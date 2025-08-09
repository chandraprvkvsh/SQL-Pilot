import os
import json
import asyncio
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from dataclasses import dataclass
from enum import Enum
import tkinter as tk
from tkinter import filedialog
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import operator
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    REQUIRES_CONSENT = "requires_consent"

@dataclass
class DatabaseOperation:
    operation_type: str
    table: str
    parameters: Dict[str, Any]
    requires_consent: bool = False

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_query: str
    parsed_operations: List[DatabaseOperation]
    execution_results: List[Dict[str, Any]]
    current_user: Optional[str]
    is_authenticated: bool
    error_message: Optional[str]
    requires_human_approval: bool
    execution_status: ExecutionStatus
    conversation_context: Dict[str, Any]
    database_path: Optional[str]

class MCPDatabaseAgent:
    def __init__(self, 
                 mcp_server_url: str = "ws://localhost:8000",
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4o"):
        
        self.mcp_server_url = mcp_server_url
        self.websocket = None
        
        if llm_provider == "openai":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        elif llm_provider == "anthropic":
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        
        builder.add_node("query_analyzer", self._analyze_query)
        builder.add_node("authenticator", self._handle_authentication)
        builder.add_node("operation_planner", self._plan_operations)
        builder.add_node("consent_checker", self._check_consent)
        builder.add_node("executor", self._execute_operations)
        builder.add_node("response_formatter", self._format_response)
        builder.add_node("error_handler", self._handle_errors)
        
        builder.add_edge(START, "query_analyzer")
        builder.add_conditional_edges(
            "query_analyzer",
            self._route_after_analysis,
            {
                "authenticate": "authenticator",
                "plan": "operation_planner",
                "error": "error_handler"
            }
        )
        
        builder.add_conditional_edges(
            "authenticator",
            self._route_after_auth,
            {
                "success": "operation_planner",
                "error": "error_handler"
            }
        )
        
        builder.add_conditional_edges(
            "operation_planner",
            self._route_after_planning,
            {
                "check_consent": "consent_checker",
                "execute": "executor",
                "error": "error_handler"
            }
        )
        
        builder.add_conditional_edges(
            "consent_checker",
            self._route_after_consent,
            {
                "execute": "executor",
                "require_approval": "response_formatter",
                "error": "error_handler"
            }
        )
        
        builder.add_edge("executor", "response_formatter")
        builder.add_edge("response_formatter", END)
        builder.add_edge("error_handler", END)
        
        return builder.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
    
    async def _connect_to_mcp(self):
        if not self.websocket:
            try:
                self.websocket = await websockets.connect(self.mcp_server_url)
                logger.info(f"Connected to MCP server at {self.mcp_server_url}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                raise
    
    async def _call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await self._connect_to_mcp()
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = """You are an expert SQL database assistant. Analyze the user's natural language query and determine:
        1. What database operation they want to perform (read, insert, update, delete, create table, etc.)
        2. Which table(s) are involved
        3. What specific data or conditions are mentioned
        4. Whether authentication is needed
        
        Respond with a JSON object containing:
        - operation_type: the type of operation
        - requires_auth: boolean indicating if authentication is needed
        - confidence: how confident you are in the analysis (0-1)
        - next_action: "authenticate", "plan", or "error"
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            analysis = json.loads(response.content)
            return {
                "messages": [AIMessage(content=f"Analyzed query: {analysis}")],
                "conversation_context": analysis,
                "requires_human_approval": False
            }
        except json.JSONDecodeError:
            return {
                "messages": [AIMessage(content="Failed to analyze query")],
                "error_message": "Query analysis failed",
                "execution_status": ExecutionStatus.ERROR
            }
    
    async def _handle_authentication(self, state: AgentState) -> Dict[str, Any]:
        if state.get("is_authenticated"):
            return {"messages": [AIMessage(content="Already authenticated")]}
        
        auth_result = await self._call_mcp_tool("authenticate", {
            "username": "admin",
            "password": "admin123"
        })
        
        if "error" not in auth_result:
            return {
                "messages": [AIMessage(content="Authentication successful")],
                "is_authenticated": True,
                "current_user": "admin"
            }
        else:
            return {
                "messages": [AIMessage(content="Authentication failed")],
                "error_message": auth_result.get("error", "Authentication failed"),
                "execution_status": ExecutionStatus.ERROR
            }
    
    async def _plan_operations(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = """Based on the user query and analysis, create a detailed execution plan.
        Generate a list of DatabaseOperation objects with:
        - operation_type: specific MCP tool name (list_tables, read_data, insert_data, etc.)
        - table: table name if applicable
        - parameters: dict of parameters for the operation
        - requires_consent: boolean for destructive operations
        
        Return as JSON array of operations.
        """
        
        context = state.get("conversation_context", {})
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {state['user_query']}\nContext: {context}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            operations_data = json.loads(response.content)
            operations = [DatabaseOperation(**op) for op in operations_data]
            
            return {
                "messages": [AIMessage(content=f"Planned {len(operations)} operations")],
                "parsed_operations": operations
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content="Failed to plan operations")],
                "error_message": f"Planning failed: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }
    
    async def _check_consent(self, state: AgentState) -> Dict[str, Any]:
        destructive_ops = [op for op in state["parsed_operations"] if op.requires_consent]
        
        if not destructive_ops:
            return {"messages": [AIMessage(content="No consent required")]}
        
        consent_results = []
        for op in destructive_ops:
            result = await self._call_mcp_tool("grant_consent", {
                "tool_name": op.operation_type,
                "table": op.table
            })
            consent_results.append(result)
        
        if all("error" not in result for result in consent_results):
            return {
                "messages": [AIMessage(content="Consent granted for all operations")],
                "execution_status": ExecutionStatus.SUCCESS
            }
        else:
            return {
                "messages": [AIMessage(content="Human approval required for destructive operations")],
                "requires_human_approval": True,
                "execution_status": ExecutionStatus.REQUIRES_CONSENT
            }
    
    async def _execute_operations(self, state: AgentState) -> Dict[str, Any]:
        results = []
        for operation in state["parsed_operations"]:
            try:
                result = await self._call_mcp_tool(
                    operation.operation_type,
                    operation.parameters
                )
                results.append({
                    "operation": operation.operation_type,
                    "result": result,
                    "status": "success" if "error" not in result else "error"
                })
            except Exception as e:
                results.append({
                    "operation": operation.operation_type,
                    "result": {"error": str(e)},
                    "status": "error"
                })
        
        return {
            "messages": [AIMessage(content=f"Executed {len(results)} operations")],
            "execution_results": results,
            "execution_status": ExecutionStatus.SUCCESS
        }
    
    async def _format_response(self, state: AgentState) -> Dict[str, Any]:
        if state.get("requires_human_approval"):
            response = "This operation requires human approval due to its destructive nature. Please confirm to proceed."
        elif state.get("execution_results"):
            results = state["execution_results"]
            successful_ops = [r for r in results if r["status"] == "success"]
            failed_ops = [r for r in results if r["status"] == "error"]
            
            response_parts = []
            
            if successful_ops:
                response_parts.append(f"Successfully completed {len(successful_ops)} operations:")
                for op in successful_ops:
                    response_parts.append(f"- {op['operation']}: {op['result']}")
            
            if failed_ops:
                response_parts.append(f"Failed operations ({len(failed_ops)}):")
                for op in failed_ops:
                    response_parts.append(f"- {op['operation']}: {op['result'].get('error', 'Unknown error')}")
            
            response = "\n".join(response_parts)
        else:
            response = "Operation completed successfully."
        
        return {
            "messages": [AIMessage(content=response)]
        }
    
    async def _handle_errors(self, state: AgentState) -> Dict[str, Any]:
        error_msg = state.get("error_message", "An unknown error occurred")
        return {
            "messages": [AIMessage(content=f"Error: {error_msg}")],
            "execution_status": ExecutionStatus.ERROR
        }
    
    def _route_after_analysis(self, state: AgentState) -> str:
        context = state.get("conversation_context", {})
        if context.get("requires_auth") and not state.get("is_authenticated"):
            return "authenticate"
        elif context.get("next_action") == "plan":
            return "plan"
        else:
            return "error"
    
    def _route_after_auth(self, state: AgentState) -> str:
        return "success" if state.get("is_authenticated") else "error"
    
    def _route_after_planning(self, state: AgentState) -> str:
        operations = state.get("parsed_operations", [])
        if not operations:
            return "error"
        
        destructive_ops = [op for op in operations if op.requires_consent]
        return "check_consent" if destructive_ops else "execute"
    
    def _route_after_consent(self, state: AgentState) -> str:
        if state.get("requires_human_approval"):
            return "require_approval"
        elif state.get("execution_status") == ExecutionStatus.SUCCESS:
            return "execute"
        else:
            return "error"
    
    async def invoke(self, user_query: str, thread_id: str = "default", database_path: str = None) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "parsed_operations": [],
            "execution_results": [],
            "current_user": None,
            "is_authenticated": False,
            "error_message": None,
            "requires_human_approval": False,
            "execution_status": ExecutionStatus.PENDING,
            "conversation_context": {},
            "database_path": database_path
        }
        
        try:
            result = await self.graph.ainvoke(initial_state, config)
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"Agent execution failed: {str(e)}"
    
    async def cleanup(self):
        if self.websocket:
            await self.websocket.close()
