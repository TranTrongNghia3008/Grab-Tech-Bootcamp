import os
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Import Schemas
from src.schemas.tool_schemas import (
    ProfileTableSchema, GetSampleDataSchema, ExecuteSQLSchema,
    VisualizeDataSchema, ExplainSQLSchema, GetTableSchemaInput,
    FindRelatedTablesInput, GetColumnValueDistributionInput, GetTableIndexesInput
)
# Import Tool Functions
from src.tools.database_tools import (
    list_tables, profile_table, get_table_schema, find_related_tables,
    get_column_value_distribution, get_table_indexes
)
from src.tools.query_tools import get_sample_data, execute_sql, explain_sql_query
from src.tools.visualization_tool import visualize_data

# Import Config
from src.config import OPENAI_API_KEY, AGENT_MODEL, AGENT_TEMPERATURE

def create_agent_executor() -> AgentExecutor:
    """Initializes LLM, Memory, Tools, Prompt, and creates the Agent Executor."""

    # --- LLM ---
    llm = ChatOpenAI(
        model=AGENT_MODEL,
        temperature=AGENT_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        # Add streaming=True if desired and handle in main loop
    )

    # --- Memory ---
    # Consider ConversationSummaryBufferMemory for longer conversations
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- Tool Definitions ---
    list_tables_tool = Tool(
        name="list_tables",
        func=list_tables, # Pass the function reference
        description="List all table names in the connected PostgreSQL database as a JSON string. No input required."
    )

    profile_table_tool = StructuredTool.from_function(
        func=profile_table,
        name="profile_table",
        description="Generate profiling statistics (row count, columns stats like nulls, distinct values, numeric summaries, string lengths) for a specific table. Returns profile as a JSON string.",
        args_schema=ProfileTableSchema
    )

    get_sample_data_tool = StructuredTool.from_function(
        func=get_sample_data,
        name="get_sample_data",
        description="Fetches a small sample of rows (default 10, max configurable) from a specified table name to inspect actual data values. Returns data as a formatted string.",
        args_schema=GetSampleDataSchema
    )

    execute_sql_tool = StructuredTool.from_function(
        func=execute_sql,
        name="execute_sql",
        description=(
            "Executes a given SQL query against the database. "
            "CRITICAL: Use with extreme caution. Only execute queries that have been carefully reviewed and approved, especially UPDATE, DELETE, or ALTER statements, as they modify data. "
            "This tool does NOT return data from SELECT queries (use get_sample_data or visualize_data instead); it only confirms execution for SELECTs. For other statements, it returns rows affected or an error message."
        ),
        args_schema=ExecuteSQLSchema
    )

    visualize_data_tool = StructuredTool.from_function(
        func=visualize_data,
        name="visualize_data",
        description=(
            "Generates a plot from data obtained via a SQL query and saves it as a PNG file. "
            "Requires: a valid SELECT SQL query string (`sql_query`), the type of chart (`chart_type`: 'bar', 'line', 'scatter', 'pie', 'hist'), "
            "the column for the x-axis (`x_col`), and potentially the column for the y-axis (`y_col` - required for bar, line, scatter, pie). "
            "Optionally takes a `color_col` for hue/grouping. "
            "Returns the file path of the saved plot or an error message. Ensure the SQL query selects the necessary columns."
        ),
        args_schema=VisualizeDataSchema
    )

    explain_sql_query_tool = StructuredTool.from_function(
        func=explain_sql_query,
        name="explain_sql_query",
        description=(
            "Analyzes a given SQL query using PostgreSQL's EXPLAIN ANALYZE (with detailed options like COSTS, BUFFERS) to show the query execution plan and performance statistics. "
            "This helps in understanding how a query runs and identifying potential optimizations. "
            "Input is the SQL query string (primarily intended for SELECT queries). "
            "Returns the query plan (usually as JSON) or an error message. Note: EXPLAIN ANALYZE *executes* the query, but changes are rolled back by this tool."
        ),
        args_schema=ExplainSQLSchema
    )

    get_table_schema_tool = StructuredTool.from_function(
        func=get_table_schema,
        name="get_table_schema",
        description="Retrieves detailed schema information for a specific table, including columns (name, type, nullable), primary key, foreign keys, and indexes. Returns schema as a JSON string.",
        args_schema=GetTableSchemaInput
    )

    find_related_tables_tool = StructuredTool.from_function(
        func=find_related_tables,
        name="find_related_tables",
        description="Finds and lists tables directly related to the specified table via foreign key constraints (both outgoing and incoming references across schemas). Returns relationships as a JSON string. Note: May be slow on databases with very many tables.",
        args_schema=FindRelatedTablesInput
    )

    get_column_value_distribution_tool = StructuredTool.from_function(
        func=get_column_value_distribution,
        name="get_column_value_distribution",
        description="Analyzes value distribution for a specific column. Returns top N value counts for categorical columns, or key percentiles/stats for numeric columns. Returns analysis as a JSON string.",
        args_schema=GetColumnValueDistributionInput
    )

    get_table_indexes_tool = StructuredTool.from_function(
        func=get_table_indexes,
        name="get_table_indexes",
        description="Retrieves information about indexes defined on a specific table. Returns index list as a JSON string.",
        args_schema=GetTableIndexesInput
    )

    tools = [
        list_tables_tool,
        get_table_schema_tool,
        find_related_tables_tool,
        profile_table_tool,
        get_column_value_distribution_tool,
        get_sample_data_tool,
        get_table_indexes_tool,
        explain_sql_query_tool,
        visualize_data_tool,
        execute_sql_tool, # Keep potentially dangerous tools last or add more warnings
    ]

    # --- Prompt Template ---
    # Keep your detailed prompt, ensure variable names match memory key and input key
    # Using f-string for multiline clarity
    system_prompt = f"""You are an AI assistant expert in interacting with a PostgreSQL database. Your primary goal is to help users explore, analyze, query, visualize, and manage their database effectively and safely.

**Your Core Capabilities & Tools:**

You have access to the following tools to interact with the database:

1.  **`list_tables`**: List all available table names (returns JSON string).
2.  **`get_table_schema`**: Retrieve detailed structure for a specific table (columns, types, PK, FKs, indexes) (returns JSON string). Use this to understand table structure *before* writing complex queries.
3.  **`find_related_tables`**: Identify tables connected via Foreign Keys to a specific table (returns JSON string). Essential for planning JOIN operations.
4.  **`profile_table`**: Get statistics for a table's columns (null rate, distinct count, numeric stats, avg string length) (returns JSON string). Useful for initial data quality assessment.
5.  **`get_column_value_distribution`**: Analyze the distribution of values in a specific column (top counts for categorical, percentiles for numeric) (returns JSON string). Use this for deeper data understanding.
6.  **`get_sample_data`**: Fetch a small sample of rows from a table to inspect actual data values (returns string).
7.  **`get_table_indexes`**: List the indexes defined on a specific table (returns JSON string). Useful for performance context.
8.  **`explain_sql_query`**: Analyze a query's execution plan using `EXPLAIN ANALYZE` (returns JSON string). Use this to understand query performance.
9.  **`visualize_data`**: Generate plots (bar, line, scatter, pie, hist) from data retrieved via a SQL query and save them to a file (returns file path string). **After creating a plot, you MUST explain it.**
10. **`execute_sql`**: Execute a given SQL query. **USE WITH EXTREME CAUTION.** (returns status/rows affected string).

**Your Workflow & Process:**

1.  **Understand Goal:** Analyze the user's request. Ask clarifying questions if needed.
2.  **Gather Context (Information Tools):** Use `list_tables`, `get_table_schema`, `find_related_tables`, `profile_table`, `get_column_value_distribution`, `get_sample_data`, `get_table_indexes` as needed BEFORE generating complex SQL or performing actions. Assume results from these tools are JSON strings unless otherwise noted; parse them mentally to inform your next steps.
3.  **Formulate Plan & Generate SQL/Viz Specs:** Create a step-by-step plan. Generate precise PostgreSQL SQL. Determine `visualize_data` parameters.
4.  **Present Plan & Seek Confirmation (ESPECIALLY for Modifications):** Explain your plan. **CRITICAL SAFETY STEP:** If using `execute_sql` for `UPDATE`, `INSERT`, `DELETE`, `ALTER`, `DROP`, etc., **MUST** present the exact SQL and get explicit user confirmation before execution. Briefly describe intended `SELECT` or visualization actions.
5.  **Execute Approved Actions (Action Tools):** Use `execute_sql`, `visualize_data`, `explain_sql_query` after approval. Report outcome of each step if modifying data.
6.  **Report Results & Explain Visualizations:** Present tool output clearly.
    * For `execute_sql` (non-SELECT): Report success/rows affected or error.
    * **For `visualize_data`:** Report the file path. **Then, provide a brief textual summary interpreting the chart** (type, axes, main message, trends, insights).
    * For `explain_sql_query`: Summarize the key findings from the query plan JSON.
    * For informational tools: Summarize the relevant information from the JSON output.
7.  **Chart Image Analysis & Recommendation (If User Provides Image):**
    * If the user input includes image data:
        * **Analyze & Describe:** Examine the chart. Identify type, axes, title, legend, main patterns/trends.
        * **Explain & Insights:** What does the chart show? What are the key takeaways visible?
        * **Critique Effectiveness:** Is it clear? Appropriate type? Cluttered? Misleading? Areas for improvement?
        * **Recommend Alternatives:** Suggest better chart types or improvements (e.g., "A line chart for time series," "Log scale for wide ranges," "Fewer pie slices"). Explain *why*.
        * **Connect to Data (If Possible):** Link visual analysis to known data context if available. Otherwise, base purely on the image.

**Important Guidelines:**

* **Prioritize Safety:** Never execute data modification queries (`UPDATE`, `INSERT`, `DELETE`, `ALTER`, `DROP`) without explicit user confirmation of the exact SQL. Avoid generating dangerous queries like `DROP TABLE` without strong justification and multiple confirmations. Be cautious with queries lacking `WHERE` clauses.
* **Use Tools Intelligently:** Don't guess; use schema/profiling tools first. Use `explain_sql_query` before suggesting complex index changes.
* **Be Precise:** Generate correct PostgreSQL syntax. Use quotes (`"`) for identifiers if necessary (e.g., mixed case, keywords).
* **Handle Errors:** Report tool errors and adapt your plan. Inform the user if an action failed.
* **Stay Focused:** Stick to the database interaction task based on the user's request.

Your goal is to be reliable, knowledgeable, and safe. Think step-by-step, gather info, confirm actions, communicate clearly, and explain visualizations.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            # The 'input' variable can handle text or a list of content blocks (text, image)
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # --- Agent ---
    # create_openai_tools_agent is suitable for models supporting OpenAI function/tool calling
    agent = create_openai_tools_agent(llm, tools, prompt)

    # --- Agent Executor ---
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True, # Set to False for cleaner production output
        handle_parsing_errors=True, # Provides default handling for parsing errors
        # max_iterations=15, # Optional: Limit cycles to prevent runaways
        # return_intermediate_steps=True # Optional: If you want to see thought process
    )

    return agent_executor