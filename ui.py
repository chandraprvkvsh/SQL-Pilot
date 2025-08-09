import gradio as gr
import asyncio
import os
import sqlite3
import shutil
from agent import MCPDatabaseAgent
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseAgentUI:
    def __init__(self):
        self.agent = MCPDatabaseAgent(
            mcp_server_url="ws://localhost:8000",
            llm_provider="openai",
            model_name="gpt-4o"
        )
        self.conversation_history = []
        self.current_database = None
    
    def validate_database_file(self, file_path: str) -> tuple:
        if not file_path:
            return False, "No file selected"
        
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if not file_path.lower().endswith('.db'):
            return False, "File must be a SQLite database (.db extension)"
        
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            if not tables:
                return True, f"Database loaded successfully (empty database)"
            else:
                table_names = [table[0] for table in tables]
                return True, f"Database loaded successfully with tables: {', '.join(table_names)}"
                
        except sqlite3.Error as e:
            return False, f"Invalid SQLite database: {str(e)}"
    
    def load_database(self, file):
        if file is None:
            return None, "Please select a database file first", ""
        
        try:
            file_path = file.name
            is_valid, message = self.validate_database_file(file_path)
            
            if is_valid:
                self.current_database = file_path
                return file_path, message, f"Current Database: {os.path.basename(file_path)}"
            else:
                return None, message, ""
                
        except Exception as e:
            return None, f"Error loading database: {str(e)}", ""
    
    def create_sample_database(self):
        temp_dir = tempfile.gettempdir()
        sample_db_path = os.path.join(temp_dir, "sample_database.db")
        
        try:
            conn = sqlite3.connect(sample_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            cursor.execute("INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)", ("John Doe", "john@example.com"))
            cursor.execute("INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)", ("Jane Smith", "jane@example.com"))
            cursor.execute("INSERT OR IGNORE INTO posts (user_id, title, content) VALUES (?, ?, ?)", 
                          (1, "Welcome Post", "This is a sample post"))
            
            conn.commit()
            conn.close()
            
            self.current_database = sample_db_path
            return sample_db_path, "Sample database created successfully with users and posts tables", f"Current Database: sample_database.db"
            
        except Exception as e:
            return None, f"Error creating sample database: {str(e)}", ""
    
    async def process_query(self, message: str, history: list) -> tuple:
        if not self.current_database:
            history.append([message, "Please load a database file first using the 'Upload Database' section above."])
            return history, ""
        
        try:
            history.append([message, None])
            
            response = await self.agent.invoke(message, thread_id="gradio_session", database_path=self.current_database)
            
            history[-1][1] = response
            
            self.conversation_history.append({
                "user": message,
                "agent": response
            })
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            history[-1][1] = error_msg
            return history, ""
    
    def clear_history(self):
        self.conversation_history = []
        return []
    
    def get_example_queries(self):
        return [
            "Show me all tables in the database",
            "List all users in the users table",
            "Add a new user with name 'Alice' and email 'alice@example.com'",
            "Update the user with id 1 to have name 'Bob'",
            "Delete all posts by user id 2",
            "Create a new table called 'products' with columns id, name, and price",
            "Show me the schema of the posts table",
            "Count how many users are in the database",
            "Show me all posts with their user names"
        ]
    
    def create_interface(self):
        with gr.Blocks(
            title="MCP Database Agent",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            .database-status {
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                background-color: #f0f0f0;
            }
            """
        ) as demo:
            
            gr.Markdown("MCP Database Agent")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("Database Management")
                    
                    database_file = gr.File(
                        label="Upload SQLite Database (.db file)",
                        file_types=[".db"],
                        file_count="single"
                    )
                    
                    with gr.Row():
                        load_btn = gr.Button("Load Database", variant="primary")
                        sample_btn = gr.Button("Create Sample DB", variant="secondary")
                    
                    database_status = gr.Textbox(
                        label="Database Status",
                        interactive=False,
                        placeholder="No database loaded"
                    )
                    
                    current_db_display = gr.Textbox(
                        label="Current Database",
                        interactive=False,
                        placeholder="None"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("Chat with Database")
                    
                    chatbot = gr.Chatbot(
                        label="Database Agent Chat",
                        height=400,
                        show_label=False,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Query",
                            placeholder="Ask me anything about the database...",
                            container=False,
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Example Queries")
                    
                    example_queries = self.get_example_queries()
                    for i, query in enumerate(example_queries):
                        gr.Button(
                            query,
                            variant="outline",
                            size="sm"
                        ).click(
                            fn=lambda q=query: q,
                            outputs=msg
                        )
            
            def process_message_wrapper(message, history):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.process_query(message, history)
                    )
                finally:
                    loop.close()
            
            load_btn.click(
                fn=self.load_database,
                inputs=[database_file],
                outputs=[database_file, database_status, current_db_display]
            )
            
            sample_btn.click(
                fn=self.create_sample_database,
                outputs=[database_file, database_status, current_db_display]
            )
            
            submit_btn.click(
                fn=process_message_wrapper,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                fn=process_message_wrapper,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                fn=self.clear_history,
                outputs=chatbot
            )
        
        return demo
    
    def launch(self, **kwargs):
        demo = self.create_interface()
        demo.launch(**kwargs)

if __name__ == "__main__":
    ui = DatabaseAgentUI()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
