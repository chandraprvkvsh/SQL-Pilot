import os
import json
import asyncio
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from dataclasses import dataclass
from enum import Enum
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import operator
import logging
from fastmcp import Client
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY

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
                 mcp_server_url: str = "http://localhost:8000/mcp",
                 llm_provider: str = "openai",
                 model_name: str = "gpt-5-mini",
                 checkpoint_db_path: str = ":memory:"):
        
        self.mcp_server_url = mcp_server_url
        self._client = None
        self.checkpoint_db_path = checkpoint_db_path
        
        if llm_provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in config")
            self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=OPENAI_API_KEY)
        elif llm_provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set in config")
            self.llm = ChatAnthropic(model=model_name, temperature=0, api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            
        self.store = InMemoryStore()
        self.graph = None
        self._checkpointer_context = None
        self._checkpointer = None
        self._is_initialized = False
        self._authentication_state = {"is_authenticated": False, "current_user": None}

    async def get_client(self):
        """Get or create MCP client with proper connection handling"""
        if self._client is None:
            self._client = Client(self.mcp_server_url)
        return self._client

    async def initialize(self):
        """Initialize the agent graph and checkpointer"""
        if self._is_initialized:
            return

        self._checkpointer_context = AsyncSqliteSaver.from_conn_string(self.checkpoint_db_path)
        self._checkpointer = await self._checkpointer_context.__aenter__()

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
            {"authenticate": "authenticator", "plan": "operation_planner", "error": "error_handler"}
        )
        
        builder.add_conditional_edges(
            "authenticator",
            self._route_after_auth,
            {"success": "operation_planner", "error": "error_handler"}
        )
        
        builder.add_conditional_edges(
            "operation_planner",
            self._route_after_planning,
            {"check_consent": "consent_checker", "execute": "executor", "error": "error_handler"}
        )
        
        builder.add_conditional_edges(
            "consent_checker",
            self._route_after_consent,
            {"execute": "executor", "require_approval": "response_formatter", "error": "error_handler"}
        )
        
        builder.add_edge("executor", "response_formatter")
        builder.add_edge("response_formatter", END)
        builder.add_edge("error_handler", END)

        self.graph = builder.compile(checkpointer=self._checkpointer, store=self.store)
        self._is_initialized = True
        logger.info("Agent initialized successfully")

    async def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the user query and determine next action"""
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
- table_involved: table name if applicable
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ]

        try:
            response = await self.llm.ainvoke(messages)
            analysis = json.loads(response.content)
            
            return {
                "messages": [AIMessage(content=f"Analyzed query: {analysis}")],
                "conversation_context": analysis,
                "requires_human_approval": False
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return {
                "messages": [AIMessage(content="Failed to analyze query")],
                "error_message": "Query analysis failed - invalid JSON response",
                "execution_status": ExecutionStatus.ERROR
            }
        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return {
                "messages": [AIMessage(content="Failed to analyze query")],
                "error_message": f"Query analysis failed: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }

    async def _handle_authentication(self, state: AgentState) -> Dict[str, Any]:
        """Handle authentication with the MCP server"""
        if self._authentication_state["is_authenticated"]:
            return {
                "messages": [AIMessage(content="Already authenticated")],
                "is_authenticated": True,
                "current_user": self._authentication_state["current_user"]
            }

        try:
            client = await self.get_client()
            async with client:
                auth_result = await client.call_tool("authenticate", {
                    "username": "admin",
                    "password": "admin123"
                })

                if "error" not in str(auth_result):
                    self._authentication_state["is_authenticated"] = True
                    self._authentication_state["current_user"] = "admin"
                    return {
                        "messages": [AIMessage(content="Authentication successful")],
                        "is_authenticated": True,
                        "current_user": "admin"
                    }
                else:
                    return {
                        "messages": [AIMessage(content="Authentication failed")],
                        "error_message": f"Authentication failed: {auth_result}",
                        "execution_status": ExecutionStatus.ERROR
                    }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                "messages": [AIMessage(content="Authentication failed")],
                "error_message": f"Authentication error: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }

    async def _plan_operations(self, state: AgentState) -> Dict[str, Any]:
        """Plan database operations based on the query analysis"""
        system_prompt = """Based on the user query and analysis, create a detailed execution plan.

Generate a list of DatabaseOperation objects with:
- operation_type: specific MCP tool name (list_tables, read_data, insert_data, etc.)
- table: table name if applicable
- parameters: dict of parameters for the operation
- requires_consent: boolean for destructive operations (insert, update, delete, create, drop)

Return as JSON array of operations.

Available tools:
- list_tables: List all tables
- describe_table: Get table schema
- read_data: Select data from table
- insert_data: Insert new row
- update_data: Update existing rows
- delete_data: Delete rows
- create_table: Create new table
- drop_table: Drop table
"""

        context = state.get("conversation_context", {})
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {state['user_query']}\nContext: {context}")
        ]

        try:
            response = await self.llm.ainvoke(messages)
            operations_data = json.loads(response.content)
            
            operations = []
            for op_data in operations_data:
                op_type = op_data.get("operation_type", "")
                requires_consent = op_type in ["insert_data", "update_data", "delete_data", "create_table", "drop_table"]
                
                operations.append(DatabaseOperation(
                    operation_type=op_data.get("operation_type", ""),
                    table=op_data.get("table", ""),
                    parameters=op_data.get("parameters", {}),
                    requires_consent=requires_consent
                ))

            return {
                "messages": [AIMessage(content=f"Planned {len(operations)} operations")],
                "parsed_operations": operations
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planning response: {e}")
            return {
                "messages": [AIMessage(content="Failed to plan operations")],
                "error_message": f"Planning failed - invalid JSON: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }
        except Exception as e:
            logger.error(f"Planning error: {e}")
            return {
                "messages": [AIMessage(content="Failed to plan operations")],
                "error_message": f"Planning failed: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }

    async def _check_consent(self, state: AgentState) -> Dict[str, Any]:
        """Check consent for destructive operations"""
        destructive_ops = [op for op in state["parsed_operations"] if op.requires_consent]
        
        if not destructive_ops:
            return {"messages": [AIMessage(content="No consent required")]}

        try:
            client = await self.get_client()
            consent_results = []
            
            async with client:
                for op in destructive_ops:
                    result = await client.call_tool("grant_consent", {
                        "tool_name": op.operation_type,
                        "table": op.table
                    })
                    consent_results.append(result)

            if all("error" not in str(result) for result in consent_results):
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
        except Exception as e:
            logger.error(f"Consent checking error: {e}")
            return {
                "messages": [AIMessage(content="Consent checking failed")],
                "error_message": f"Consent checking failed: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }

    async def _execute_operations(self, state: AgentState) -> Dict[str, Any]:
        """Execute the planned database operations"""
        results = []
        
        try:
            client = await self.get_client()
            async with client:
                for operation in state["parsed_operations"]:
                    try:
                        result = await client.call_tool(
                            operation.operation_type,
                            operation.parameters
                        )
                        
                        results.append({
                            "operation": operation.operation_type,
                            "result": result,
                            "status": "success" if "error" not in str(result) else "error"
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
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "messages": [AIMessage(content="Operation execution failed")],
                "error_message": f"Execution failed: {str(e)}",
                "execution_status": ExecutionStatus.ERROR
            }

    async def _format_response(self, state: AgentState) -> Dict[str, Any]:
        """Format the final response based on execution results"""
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
                    error_msg = op['result'].get('error', 'Unknown error') if isinstance(op['result'], dict) else str(op['result'])
                    response_parts.append(f"- {op['operation']}: {error_msg}")
            
            response = "\n".join(response_parts)
        else:
            response = "Operation completed successfully."

        return {
            "messages": [AIMessage(content=response)]
        }

    async def _handle_errors(self, state: AgentState) -> Dict[str, Any]:
        """Handle errors and format error response"""
        error_msg = state.get("error_message", "An unknown error occurred")
        return {
            "messages": [AIMessage(content=f"Error: {error_msg}")],
            "execution_status": ExecutionStatus.ERROR
        }

    def _route_after_analysis(self, state: AgentState) -> str:
        """Route after query analysis"""
        context = state.get("conversation_context", {})
        if context.get("requires_auth") and not self._authentication_state["is_authenticated"]:
            return "authenticate"
        elif context.get("next_action") == "plan":
            return "plan"
        else:
            return "error"

    def _route_after_auth(self, state: AgentState) -> str:
        """Route after authentication"""
        return "success" if state.get("is_authenticated") else "error"

    def _route_after_planning(self, state: AgentState) -> str:
        """Route after operation planning"""
        operations = state.get("parsed_operations", [])
        if not operations:
            return "error"
        
        destructive_ops = [op for op in operations if op.requires_consent]
        return "check_consent" if destructive_ops else "execute"

    def _route_after_consent(self, state: AgentState) -> str:
        """Route after consent checking"""
        if state.get("requires_human_approval"):
            return "require_approval"
        elif state.get("execution_status") == ExecutionStatus.SUCCESS:
            return "execute"
        else:
            return "error"

    async def invoke(self, user_query: str, thread_id: str = "default", database_path: str = None) -> str:
        """Invoke the agent with a user query"""
        try:
            await self.initialize()
            
            config = {"configurable": {"thread_id": thread_id}}
            
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "parsed_operations": [],
                "execution_results": [],
                "current_user": self._authentication_state["current_user"],
                "is_authenticated": self._authentication_state["is_authenticated"],
                "error_message": None,
                "requires_human_approval": False,
                "execution_status": ExecutionStatus.PENDING,
                "conversation_context": {},
                "database_path": database_path
            }

            result = await self.graph.ainvoke(initial_state, config)
            return result["messages"][-1].content
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"Agent execution failed: {str(e)}"

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self._checkpointer_context is not None:
                await self._checkpointer_context.__aexit__(None, None, None)
                self._checkpointer_context = None
                self._checkpointer = None
            
            if self._client is not None:
                self._client = None
                
            self._is_initialized = False
            logger.info("Agent cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
