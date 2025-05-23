import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import datetime
import json # For potentially pretty-printing dictionary results
import base64
import mimetypes
import re # For parsing the image path from input


# --- SQLAlchemy Imports ---

from sqlalchemy import create_engine, inspect, text, MetaData
from sqlalchemy.engine.reflection import Inspector # Fixed import path
from sqlalchemy import func # For SQL functions like COUNT, percentile_cont
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError, OperationalError


# --- Langchain Imports ---
# Use StructuredTool for tools with multiple/typed arguments
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# --- Pydantic for Argument Schemas ---
from pydantic.v1 import BaseModel, Field # Use v1 for compatibility if needed, or just pydantic

load_dotenv()

# --- DATABASE CONNECTION SETUP ---
# ... (keep your existing db connection setup)
db_user = os.getenv("PG_USER")
db_password = os.getenv("PG_PASSWORD")
db_host = os.getenv("PG_HOST")
db_port = os.getenv("PG_PORT", "5432")
db_name = os.getenv("PG_DATABASE")
if not all([db_user, db_password, db_host, db_name]):
    raise ValueError("Missing one or more PostgreSQL connection environment variables")
db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_uri)
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Database connection successful.")
except Exception as e:
    print(f"Database connection failed: {e}")
    exit()

# chart for plotting
PLOT_DIR = Path("./plots")
PLOT_DIR.mkdir(exist_ok=True)

# --- Image Directory for local chart visualisation ---
LOCAL_CHART_DIR = Path("./local_chart_images")
LOCAL_CHART_DIR.mkdir(exist_ok=True) # Create if it doesn't exist

# --- TOOL FUNCTIONS ---
# ... (keep your existing functions: list_tables, profile_table, get_sample_data, execute_sql, visualize_data, explain_sql_query)
# Make sure the function signatures match what the Pydantic models will expect

def encode_image_to_base64(image_path: Path) -> tuple[str | None, str | None]:
    """Reads an image file, encodes it to base64, and determines MIME type."""
    try:
        if not image_path.is_file():
            print(f"Error: Image file not found at {image_path}")
            return None, None

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            print(f"Error: Invalid image MIME type: {mime_type} for {image_path}")
            return None, None

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type

    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None, None

def table_exists(inspector: Inspector, table_name: str, schema: str | None = None) -> bool:
    """Checks if a table exists using the inspector."""
    try:
        # inspector.get_columns() raises NoSuchTableError if table doesn't exist
        inspector.get_columns(table_name, schema=schema)
        return True
    except NoSuchTableError:
        return False
    except Exception as e:
        print(f"Error checking if table {table_name} exists: {e}")
        return False # Assume false on other errors
def get_table_schema(table_name: str) -> str:
    """
    Retrieves detailed schema information for a specific table.
    Includes columns (name, type, nullable, default), primary key,
    foreign keys, and indexes.
    Input is the table name (string).
    Returns a description of the schema as a JSON string or an error message.
    """
    print(f"--- Getting schema for table: {table_name} ---")
    try:
        inspector = inspect(engine)

        if not table_exists(inspector, table_name):
             return f"Error: Table '{table_name}' not found in the database."

        schema_info = {}
        schema_info['table_name'] = table_name
        schema_info['columns'] = inspector.get_columns(table_name)
        schema_info['primary_key'] = inspector.get_pk_constraint(table_name)
        schema_info['foreign_keys'] = inspector.get_foreign_keys(table_name)
        schema_info['indexes'] = inspector.get_indexes(table_name)
        # schema_info['comment'] = inspector.get_table_comment(table_name) # Uncomment if comments are used

        # Convert complex SQLAlchemy types to strings for better serialization
        for col in schema_info.get('columns', []):
            col['type'] = str(col['type']) # Convert type object to string

        # Use json.dumps for clean, readable dictionary output
        return json.dumps(schema_info, indent=2)

    except NoSuchTableError:
         return f"Error: Table '{table_name}' not found."
    except SQLAlchemyError as db_err:
         return f"Database error while fetching schema for {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error occurred while getting schema for {table_name}: {e}"

def find_related_tables(table_name: str) -> str:
    """
    Finds and lists tables directly related to the specified table
    via foreign key constraints (both outgoing and incoming).
    Input is the table name (string).
    Returns a summary of relationships as a JSON string or an error message.
    Note: Finding incoming references might be slow on databases with many tables.
    """
    print(f"--- Finding relationships for table: {table_name} ---")
    try:
        inspector = inspect(engine)

        if not table_exists(inspector, table_name):
             return f"Error: Table '{table_name}' not found in the database."

        relationships = {
            "table": table_name,
            "references": [], # Tables referenced BY this table
            "referenced_by": [] # Tables referencing THIS table
        }

        # Outgoing: Find FKs defined in this table
        outgoing_fks = inspector.get_foreign_keys(table_name)
        for fk in outgoing_fks:
            relationships["references"].append({
                "constrained_columns": fk['constrained_columns'],
                "references_table": fk['referred_table'],
                "referenced_columns": fk['referred_columns'],
                "constraint_name": fk['name']
            })

        # Incoming: Iterate through all tables to find FKs pointing to this table
        # This can be slow on dbs with thousands of tables!
        print(f"Checking incoming references to {table_name} (this may take a moment)...")
        all_tables = inspector.get_table_names()
        for other_table in all_tables:
            if other_table == table_name:
                continue
            try:
                incoming_fks = inspector.get_foreign_keys(other_table)
                for fk in incoming_fks:
                    if fk['referred_table'] == table_name:
                        relationships["referenced_by"].append({
                            "referencing_table": other_table,
                            "constrained_columns": fk['constrained_columns'],
                            "referenced_columns": fk['referred_columns'],
                            "constraint_name": fk['name']
                        })
            except Exception as inner_e:
                 # Log error for specific table but continue checking others
                 print(f"Warning: Could not check FKs for table '{other_table}': {inner_e}")

        return json.dumps(relationships, indent=2)

    except NoSuchTableError:
         return f"Error: Table '{table_name}' not found."
    except SQLAlchemyError as db_err:
         return f"Database error while finding relationships for {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error occurred while finding relationships for {table_name}: {e}"


def get_column_value_distribution(table_name: str, column_name: str, top_n: int = 20) -> str:
    """
    Analyzes the distribution of values in a specific table column.
    For categorical columns, returns top N value counts.
    For numeric columns, returns key percentiles (min, 25%, 50%, 75%, max) and null count.
    Inputs: table_name (string), column_name (string), optional top_n (integer, default 20).
    Returns the distribution analysis as a JSON string or an error message.
    """
    print(f"--- Analyzing distribution for {table_name}.{column_name} ---")
    try:
        inspector = inspect(engine)

        # Check table existence first
        if not table_exists(inspector, table_name):
            return f"Error: Table '{table_name}' not found."

        # Get column type to determine analysis type
        columns = inspector.get_columns(table_name)
        col_info = next((col for col in columns if col['name'] == column_name), None)

        if not col_info:
            return f"Error: Column '{column_name}' not found in table '{table_name}'."

        col_type = str(col_info['type']).upper()
        distribution = {"table": table_name, "column": column_name, "type": col_type}
        quoted_table = f'"{table_name}"'
        quoted_col = f'"{column_name}"'

        with engine.connect() as conn:
             # --- Numeric Analysis (Percentiles) ---
             if any(t in col_type for t in ['INT', 'FLOAT', 'DECIMAL', 'NUMERIC', 'DOUBLE', 'REAL']):
                 distribution['analysis_type'] = 'numeric_percentiles'
                 # Note: percentile_cont might not be available in all DBs (works in PostgreSQL, SQL Server, Oracle, etc.)
                 # Ensure column is treated as numeric in SQL using ::numeric if needed (esp. for PostgreSQL)
                 percentile_sql = text(f"""
                     SELECT
                         COUNT(*) as total_count,
                         COUNT({quoted_col}) as non_null_count,
                         MIN({quoted_col}) as min,
                         MAX({quoted_col}) as max,
                         AVG({quoted_col}::numeric) as mean,
                         STDDEV_POP({quoted_col}::numeric) as stddev,
                         percentile_cont(array[0.25, 0.5, 0.75]) WITHIN GROUP (ORDER BY {quoted_col}) as percentiles
                     FROM {quoted_table}
                 """)
                 try:
                     result = conn.execute(percentile_sql).fetchone()
                     if result:
                         # fetchone returns a Row object, access by index or key (if known)
                         # Assuming standard output names or indices
                         # Need to handle potential None values (e.g., if table is empty)
                         # Convert Decimal types from avg/stddev if necessary for JSON
                         p_list = result._mapping.get('percentiles')
                         distribution.update({
                            "total_count": result._mapping.get('total_count'),
                            "non_null_count": result._mapping.get('non_null_count'),
                            "null_count": (result._mapping.get('total_count') or 0) - (result._mapping.get('non_null_count') or 0),
                            "min": result._mapping.get('min'),
                            "percentile_25": p_list[0] if p_list and len(p_list) > 0 else None,
                            "percentile_50_median": p_list[1] if p_list and len(p_list) > 1 else None,
                            "percentile_75": p_list[2] if p_list and len(p_list) > 2 else None,
                            "max": result._mapping.get('max'),
                            "mean": float(result._mapping.get('mean')) if result._mapping.get('mean') is not None else None,
                            "stddev": float(result._mapping.get('stddev')) if result._mapping.get('stddev') is not None else None,
                         })
                     else:
                         distribution['error'] = "Could not fetch numeric stats (table might be empty)."

                 except OperationalError as op_err:
                     # Handle if percentile_cont is not supported or other SQL errors
                     distribution['error'] = f"Could not calculate numeric distribution (DB function might be unsupported or query failed): {op_err}"
                     print(f"Falling back to basic stats for {table_name}.{column_name} due to error: {op_err}")
                     # Fallback to simpler stats if percentiles fail
                     try:
                        fallback_sql = text(f"SELECT COUNT(*) as total_count, COUNT({quoted_col}) as non_null_count, MIN({quoted_col}), MAX({quoted_col}), AVG({quoted_col}::numeric) FROM {quoted_table}")
                        result = conn.execute(fallback_sql).fetchone()
                        if result:
                             distribution.update({
                                "total_count": result[0],
                                "non_null_count": result[1],
                                "null_count": (result[0] or 0) - (result[1] or 0),
                                "min": result[2],
                                "max": result[3],
                                "mean": float(result[4]) if result[4] is not None else None,
                                "percentiles": "Not Available (fallback)"
                             })
                     except Exception as fallback_e:
                         distribution['error'] = f"Fallback stats also failed: {fallback_e}"


             # --- Categorical Analysis (Value Counts) ---
             elif any(t in col_type for t in ['CHAR', 'TEXT', 'VARCHAR', 'STRING', 'BOOLEAN', 'ENUM']): # Add other relevant types
                 distribution['analysis_type'] = 'categorical_counts'
                 counts_sql = text(f"""
                     SELECT {quoted_col}, COUNT(*) as count
                     FROM {quoted_table}
                     GROUP BY {quoted_col}
                     ORDER BY count DESC
                     LIMIT :limit
                 """)
                 null_count_sql = text(f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_col} IS NULL")
                 distinct_count_sql = text(f"SELECT COUNT(DISTINCT {quoted_col}) FROM {quoted_table}")

                 top_values = conn.execute(counts_sql, {'limit': top_n}).fetchall()
                 null_count = conn.execute(null_count_sql).scalar_one_or_none()
                 distinct_count = conn.execute(distinct_count_sql).scalar_one_or_none()

                 distribution['distinct_count'] = distinct_count
                 distribution['null_count'] = null_count
                 distribution['top_values'] = [dict(row._mapping) for row in top_values] # Convert Row objects to dicts

             # --- Other Types ---
             else:
                 distribution['analysis_type'] = 'other'
                 distribution['message'] = f"Distribution analysis not implemented for type '{col_type}'. Use get_sample_data or profile_table."
                 # Optionally add null count for other types
                 try:
                     null_count_sql = text(f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_col} IS NULL")
                     null_count = conn.execute(null_count_sql).scalar_one_or_none()
                     distribution['null_count'] = null_count
                 except Exception:
                      distribution['null_count'] = "Error fetching null count."

        return json.dumps(distribution, indent=2, default=str) # Use default=str to handle potential non-serializable types like Decimal

    except NoSuchTableError:
         return f"Error: Table '{table_name}' not found."
    except KeyError: # Handles case where column name is wrong after fetching columns
        return f"Error: Column '{column_name}' not found in table '{table_name}' (post-check)."
    except SQLAlchemyError as db_err:
         return f"Database error analyzing distribution for {table_name}.{column_name}: {db_err}"
    except Exception as e:
        # Catch any other unexpected error
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return f"An unexpected error occurred analyzing distribution for {table_name}.{column_name}: {e}"


def get_table_indexes(table_name: str) -> str:
    """
    Retrieves information about indexes defined on a specific table.
    Input is the table name (string).
    Returns a list of indexes as a JSON string or an error message.
    """
    print(f"--- Getting indexes for table: {table_name} ---")
    try:
        inspector = inspect(engine)

        if not table_exists(inspector, table_name):
             return f"Error: Table '{table_name}' not found in the database."

        indexes = inspector.get_indexes(table_name)
        return json.dumps(indexes, indent=2)

    except NoSuchTableError:
         return f"Error: Table '{table_name}' not found."
    except SQLAlchemyError as db_err:
         return f"Database error while fetching indexes for {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error occurred while getting indexes for {table_name}: {e}"


# --- Pydantic Schemas for New Tools ---


def list_tables(*args, **kwargs):
    """Return a list of table names in the connected database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        return f"Error listing tables: {e}"

def profile_table(table_name: str) -> dict:
    """
    Profile a single table: total_rows, per-column (null_rate, distinct_count),
    numeric (min, max, mean, std), string (avg_length). Input is the table name (string).
    """
    # ... (function implementation remains the same)
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
              return f"Error: Table '{table_name}' not found."
        cols = inspector.get_columns(table_name)
        profile = {"table": table_name, "columns": {}}
        with engine.connect() as conn:
            total_rows_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')) # Quote table name
            total = total_rows_result.scalar()

            for col in cols:
                name = col["name"]
                dtype = col["type"]
                stats = {}
                # Quote column name for safety
                quoted_name = f'"{name}"'

                # Null rate
                null_q = f'SELECT COUNT(*) FROM "{table_name}" WHERE {quoted_name} IS NULL'
                null_count = conn.execute(text(null_q)).scalar()
                stats['null_rate'] = null_count / total if total > 0 else 0

                # Distinct count
                distinct_q = f'SELECT COUNT(DISTINCT {quoted_name}) FROM "{table_name}"'
                stats['distinct_count'] = conn.execute(text(distinct_q)).scalar()

                # Numeric stats
                dtype_str = str(dtype).upper()
                if 'INT' in dtype_str or 'FLOAT' in dtype_str or 'DECIMAL' in dtype_str or 'NUMERIC' in dtype_str:
                    try:
                        num_q = f"SELECT MIN({quoted_name}), MAX({quoted_name}), AVG({quoted_name}::numeric), STDDEV_POP({quoted_name}::numeric) FROM \"{table_name}\""
                        mn, mx, avg, sd = conn.execute(text(num_q)).fetchone()
                        stats.update({ 'min': mn, 'max': mx, 'mean': float(avg) if avg else None, 'std': float(sd) if sd else None })
                    except Exception as num_e:
                        print(f"Warning: Could not calculate numeric stats for {table_name}.{name}: {num_e}")
                        stats.update({ 'min': None, 'max': None, 'mean': None, 'std': None })

                # String stats
                elif 'CHAR' in dtype_str or 'TEXT' in dtype_str or 'VARCHAR' in dtype_str:
                     try:
                         len_q = f"SELECT AVG(LENGTH({quoted_name})) FROM \"{table_name}\" WHERE {quoted_name} IS NOT NULL"
                         avg_len_result = conn.execute(text(len_q)).scalar()
                         stats['avg_length'] = float(avg_len_result) if avg_len_result else None
                     except Exception as str_e:
                         print(f"Warning: Could not calculate string stats for {table_name}.{name}: {str_e}")
                         stats['avg_length'] = None

                profile['columns'][name] = stats
        profile['total_rows'] = total
        return profile
    except Exception as e:
        return f"Error profiling table {table_name}: {e}"

def get_sample_data(table_name: str, num_rows: int = 10) -> str:
    """
    Fetches a small sample of rows (default 10, max 100) from the specified table.
    Input is the table name (string) and optionally the number of rows (integer).
    Returns the data as a string representation of a DataFrame.
    """
    # ... (function implementation remains the same)
    try:
        # Ensure num_rows is reasonable
        num_rows = max(1, min(num_rows, 100)) # Limit sample size
        query = text(f'SELECT * FROM "{table_name}" LIMIT :limit') # Use parameter binding
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={'limit': num_rows})
        return df.to_string()
    except Exception as e:
        return f"Error fetching sample data from {table_name}: {e}"

def execute_sql(sql_query: str) -> str:
    """
    Executes a given SQL query against the database.
    USE WITH EXTREME CAUTION, ESPECIALLY FOR UPDATE/DELETE/ALTER.
    For SELECT queries, it confirms execution but does not return data (use get_sample_data or visualize_data for that).
    For other query types (INSERT, UPDATE, DELETE), it returns the number of rows affected.
    Input is the SQL query string.
    """
    # ... (function implementation remains the same)
    print(f"--- Attempting to execute SQL ---\n{sql_query}\n---------------------------------")
    try:
        with engine.connect() as conn:
            with conn.begin() as transaction: # Start transaction
                try:
                    is_select = sql_query.strip().upper().startswith("SELECT")
                    result = conn.execute(text(sql_query))
                    if is_select:
                        transaction.commit()
                        return "SELECT query executed successfully. Use 'get_sample_data' or 'visualize_data' to see results."
                    else:
                        row_count = result.rowcount
                        transaction.commit() # Commit changes if no error
                        return f"Query executed successfully. Rows affected: {row_count}"
                except Exception as e:
                    print(f"Error during query execution, rolling back transaction: {e}")
                    transaction.rollback() # Rollback on error
                    return f"Error executing query: {e}. Transaction rolled back."
    except Exception as e:
        return f"Database connection or transaction error: {e}"

def visualize_data(sql_query: str, chart_type: str, x_col: str, y_col: str = None, color_col: str = None) -> str:
    """
    Executes a SQL query, fetches the data, and generates a plot (saved as a PNG file).
    Required Inputs:
        sql_query (str): The SELECT query to fetch data for plotting.
        chart_type (str): Type of chart ('bar', 'line', 'scatter', 'pie', 'hist').
        x_col (str): The column name for the X-axis (or data for hist/pie).
    Optional Input:
        y_col (str): The column name for the Y-axis (REQUIRED for 'bar', 'line', 'scatter').
        color_col (str): Column to use for color encoding (hue) in scatter or bar charts.
    Returns:
        str: Path to the saved plot image file or an error message.
    """
    # ... (function implementation: Make sure y_col check aligns with this signature)
    print(f"--- Generating visualization ---")
    print(f"Query: {sql_query}")
    print(f"Chart Type: {chart_type}, X: {x_col}, Y: {y_col}, Color: {color_col}")

    # Basic validation to ensure it's a SELECT query
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Error: visualize_data tool only works with SELECT queries."

    # Validate required y_col based on chart type
    if chart_type in ['bar', 'line', 'scatter'] and y_col is None:
        return f"Error: 'y_col' is required for chart type '{chart_type}'."

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql_query), conn)

        if df.empty:
            return "Error: Query returned no data to plot."

        # Determine required columns based on chart type and provided args
        required_cols = [x_col]
        if y_col and chart_type not in ['hist', 'pie']:
             required_cols.append(y_col)
        if color_col:
            required_cols.append(color_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: The following required columns are missing from the query result: {', '.join(missing_cols)}"

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Generate plot based on chart_type (ensure y_col usage matches requirements)
        if chart_type == 'bar':
            sns.barplot(data=df, x=x_col, y=y_col, hue=color_col) # y_col is required
            plt.title(f'Bar Chart: {y_col} by {x_col}')
        elif chart_type == 'line':
            sns.lineplot(data=df, x=x_col, y=y_col, hue=color_col, marker='o') # y_col is required
            plt.title(f'Line Chart: {y_col} over {x_col}')
        elif chart_type == 'scatter':
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col) # y_col is required
            plt.title(f'Scatter Plot: {y_col} vs {x_col}')
        elif chart_type == 'pie':
            # Pie charts need a column for labels (using x_col) and a column for values (using y_col IF PROVIDED, otherwise assume counts)
            # This implementation assumes x_col holds categories and y_col holds numeric values for slices
            if y_col is None:
                 # If y_col is not provided, maybe we want counts of x_col? Let's require y_col for clarity.
                 # Or adjust logic: e.g., value_counts = df[x_col].value_counts()
                 return f"Error: 'y_col' specifying the numeric values for slices is required for pie chart."

            if not pd.api.types.is_numeric_dtype(df[y_col]):
                 return f"Error: Column '{y_col}' must be numeric for pie chart values."
            if df[x_col].nunique() > 15: # Limit number of slices for readability
                 return f"Error: Too many unique values in '{x_col}' for a pie chart (max 15)."

            # Use pandas plotting for pie, labels from x_col, values from y_col
            df_pie = df.set_index(x_col)
            # Ensure index (x_col) is unique if grouping wasn't done in SQL. Aggregate if needed.
            # Example simple aggregation (sum): df_pie = df.groupby(x_col)[y_col].sum()
            # Assuming SQL query already aggregated correctly:
            df_pie[y_col].plot(kind='pie', autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
            plt.ylabel('') # Hide y-label for pie
            plt.title(f'Pie Chart: Distribution of {y_col} by {x_col}')

        elif chart_type == 'hist':
             # Histogram uses only x_col (must be numeric)
            if not pd.api.types.is_numeric_dtype(df[x_col]):
                return f"Error: Column '{x_col}' must be numeric for a histogram."
            sns.histplot(data=df, x=x_col, hue=color_col, kde=True)
            plt.title(f'Histogram of {x_col}')
        else:
            plt.close()
            return f"Error: Unsupported chart type '{chart_type}'. Use 'bar', 'line', 'scatter', 'pie', or 'hist'."

        plt.xlabel(x_col)
        if chart_type in ['bar', 'line', 'scatter']: # y_col is guaranteed to exist here
             plt.ylabel(y_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{chart_type}_{x_col}_{timestamp}.png"
        save_path = PLOT_DIR / filename
        plt.savefig(save_path)
        plt.close()

        print(f"Plot saved to: {save_path}")
        return f"Plot generated successfully and saved to: {save_path}"

    except pd.errors.DatabaseError as db_err:
         return f"Error executing SQL query for visualization: {db_err}"
    except KeyError as key_err:
        return f"Error: Column not found in query results: {key_err}"
    except Exception as e:
        plt.close()
        return f"Error generating visualization: {e}"

def explain_sql_query(sql_query: str) -> str:
    """
    Analyzes a given SQL query using PostgreSQL's EXPLAIN ANALYZE command.
    Provides the query plan and execution statistics to help understand and optimize the query's performance.
    Input should be the SQL query string (preferably a SELECT statement).
    Returns the output of EXPLAIN ANALYZE as a string, or an error message.
    """
    # ... (function implementation remains the same)
    print(f"--- Explaining SQL query ---")
    print(f"Query: {sql_query}")
    if not sql_query.strip().upper().startswith("SELECT"):
         print("Warning: Running EXPLAIN ANALYZE on non-SELECT queries will execute them. Proceeding with caution.")

    try:
        explain_query = f"EXPLAIN ANALYZE {sql_query}"
        with engine.connect() as conn:
            with conn.begin() as transaction:
                try:
                    result = conn.execute(text(explain_query))
                    query_plan = result.fetchall()
                    transaction.rollback() # Always rollback after explain analyze
                    plan_output = "\n".join([row[0] for row in query_plan])
                    return f"Query Plan Analysis:\n--------------------\n{plan_output}"
                except Exception as exec_err:
                     transaction.rollback()
                     return f"Error executing EXPLAIN ANALYZE: {exec_err}. Query might be invalid."
    except Exception as e:
        return f"Database connection or transaction error during EXPLAIN ANALYZE: {e}"


# --- Pydantic Schemas for Structured Tools ---

class ProfileTableSchema(BaseModel):
    table_name: str = Field(description="The name of the table to profile.")

class GetSampleDataSchema(BaseModel):
    table_name: str = Field(description="The name of the table to fetch sample data from.")
    num_rows: int = Field(description="The number of sample rows to fetch.", default=10)

class ExplainSQLSchema(BaseModel):
     sql_query: str = Field(description="The SQL query to analyze using EXPLAIN ANALYZE.")

class VisualizeDataSchema(BaseModel):
    sql_query: str = Field(description="The SELECT query to fetch data for plotting.")
    chart_type: str = Field(description="Type of chart ('bar', 'line', 'scatter', 'pie', 'hist').")
    x_col: str = Field(description="The column name for the X-axis (or data column for hist/pie).")
    y_col: str | None = Field(description="The column name for the Y-axis (REQUIRED for 'bar', 'line', 'scatter', 'pie').", default=None) # Made optional here, validated in function
    color_col: str | None = Field(description="Optional column for color encoding (hue).", default=None)

class GetTableSchemaInput(BaseModel):
    table_name: str = Field(description="The name of the table to get the schema for.")

class FindRelatedTablesInput(BaseModel):
    table_name: str = Field(description="The name of the table to find foreign key relationships for.")

class GetColumnValueDistributionInput(BaseModel):
    table_name: str = Field(description="The name of the table containing the column.")
    column_name: str = Field(description="The name of the column to analyze the value distribution of.")
    top_n: int = Field(description="For categorical columns, the maximum number of top values to return.", default=20)

class GetTableIndexesInput(BaseModel):
    table_name: str = Field(description="The name of the table to get index information for.")

# --- TOOL DEFINITIONS ---

list_tables_tool = Tool(
    name="list_tables",
    func=list_tables,
    description="List all table names in the connected PostgreSQL database. No input required."
)

# Use StructuredTool for tools with defined arguments
profile_table_tool = StructuredTool.from_function(
    func=profile_table,
    name="profile_table",
    description="Generate profiling statistics (row count, columns stats like nulls, distinct values, numeric summaries, string lengths) for a specific table.",
    args_schema=ProfileTableSchema
)

get_sample_data_tool = StructuredTool.from_function(
    func=get_sample_data,
    name="get_sample_data",
    description="Fetches a small sample of rows (default 10, max 100) from a specified table name to inspect actual data values.",
    args_schema=GetSampleDataSchema
)


# --- Tool Definitions ---

get_table_schema_tool = StructuredTool.from_function(
    func=get_table_schema,
    name="get_table_schema",
    description="Retrieves detailed schema information for a specific table, including columns (name, type, nullable), primary key, foreign keys, and indexes. Returns schema as a JSON string.",
    args_schema=GetTableSchemaInput
)

find_related_tables_tool = StructuredTool.from_function(
    func=find_related_tables,
    name="find_related_tables",
    description="Finds and lists tables directly related to the specified table via foreign key constraints (both outgoing and incoming). Returns relationships as a JSON string. Note: May be slow on databases with very many tables.",
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

# execute_sql takes only one string, Tool is fine, but StructuredTool also works
# Using StructuredTool for consistency:
class ExecuteSQLSchema(BaseModel):
     sql_query: str = Field(description="The SQL query string to execute.")

execute_sql_tool = StructuredTool.from_function(
    func=execute_sql,
    name="execute_sql",
    description=(
        "Executes a given SQL query against the database. "
        "CRITICAL: Use with extreme caution. Only execute queries that have been carefully reviewed and approved, especially UPDATE, DELETE, or ALTER statements, as they modify data. "
        "This tool does NOT return data from SELECT queries (use get_sample_data or visualize_data instead); it only confirms execution for SELECTs. For other statements, it returns rows affected."
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
    args_schema=VisualizeDataSchema # Use the Pydantic schema here
)

explain_sql_query_tool = StructuredTool.from_function(
    func=explain_sql_query,
    name="explain_sql_query",
    description=(
        "Analyzes a given SQL query using PostgreSQL's EXPLAIN ANALYZE command to show the query execution plan and performance statistics. "
        "This helps in understanding how a query runs and identifying potential optimizations. "
        "Input is the SQL query string (primarily intended for SELECT queries). "
        "Returns the query plan as text or an error message. Note: EXPLAIN ANALYZE *executes* the query, but changes are rolled back by this tool."
    ),
    args_schema=ExplainSQLSchema
)
# --- AGENT SETUP ---

llm = ChatOpenAI( temperature=0, model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY')) # Ensure model supports tool calling well
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# List of ALL tools the agent can use (now using StructuredTool where appropriate)

tools = [
    list_tables_tool,
    profile_table_tool,
    get_sample_data_tool,
    execute_sql_tool,
    visualize_data_tool,
    explain_sql_query_tool,
    # --- Add the new tools ---
    get_table_schema_tool,
    find_related_tables_tool,
    get_column_value_distribution_tool,
    get_table_indexes_tool
]

# Prompt remains the same - it describes the capabilities
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are an AI assistant expert in interacting with a PostgreSQL database. Your primary goal is to help users explore, analyze, query, visualize, and manage their database effectively and safely.

**Your Core Capabilities & Tools:**

You have access to the following tools to interact with the database:

1.  **`list_tables`**: List all available table names.
2.  **`get_table_schema`**: Retrieve detailed structure for a specific table (columns, types, PK, FKs, indexes). Use this to understand table structure *before* writing complex queries.
3.  **`find_related_tables`**: Identify tables connected via Foreign Keys to a specific table. Essential for planning JOIN operations.
4.  **`profile_table`**: Get basic statistics for a table's columns (null rate, distinct count, numeric stats, avg string length). Useful for initial data quality assessment.
5.  **`get_column_value_distribution`**: Analyze the distribution of values in a specific column (top value counts for text/categorical, percentiles for numeric). Use this for deeper data understanding.
6.  **`get_sample_data`**: Fetch a small sample of rows from a table to inspect actual data values.
7.  **`get_table_indexes`**: List the indexes defined on a specific table. Useful for performance context.
8.  **`explain_sql_query`**: Analyze a query's execution plan using `EXPLAIN ANALYZE`. Use this to understand query performance or debug slow queries.
9.  **`visualize_data`**: Generate plots (bar, line, scatter, pie, hist) from data retrieved via a SQL query and save them to a file. Requires a SELECT query, chart type, and column specifications. **After creating a plot, you should also explain it.**
10. **`execute_sql`**: Execute a given SQL query. **USE WITH EXTREME CAUTION.**

**Your Workflow & Process:**

When responding to user requests, follow these steps diligently:

1.  **Understand the Goal:** Carefully analyze the user's request. If it's ambiguous, ask clarifying questions before proceeding.
2.  **Gather Context (Information Tools):**
    * Identify the relevant tables. If unsure, use `list_tables`.
    * **Crucially:** Before constructing complex queries or planning data modifications, use `get_table_schema` for the relevant tables.
    * Use `find_related_tables` if JOINs are needed or relationships are unclear.
    * Use `profile_table` or `get_column_value_distribution` if the request involves data quality, analysis, or understanding specific column characteristics.
    * Use `get_sample_data` sparingly if seeing actual values helps clarify structure or content.
    * Use `get_table_indexes` if performance context is relevant.
3.  **Formulate a Plan & Generate SQL/Visualization Specs (If Needed):**
    * Based on the user's goal and the gathered context, formulate a clear step-by-step plan.
    * If the plan requires executing SQL, generate the precise SQL query/queries needed for each step.
    * If visualization is requested, determine the appropriate SQL query, chart type (`bar`, `line`, `scatter`, `pie`, `hist`), and required columns (`x_col`, `y_col`, `color_col`) for the `visualize_data` tool.
    * If query explanation is needed, prepare the query for the `explain_sql_query` tool.
4.  **Present Plan & Seek Confirmation (ESPECIALLY for Modifications):**
    * Clearly explain your plan to the user.
    * **CRITICAL SAFETY STEP:** If the plan involves executing SQL that modifies data (`UPDATE`, `INSERT`, `DELETE`, `ALTER`, `DROP`, etc.) using `execute_sql`, you **MUST** present the exact SQL query to the user and get their explicit confirmation (e.g., "Yes, proceed", "Approved") before executing it.
    * For complex `SELECT` queries or visualizations generated by you, briefly describe what you intend to do before calling the tool.
5.  **Execute Approved Actions (Action Tools):**
    * Once the plan (and any necessary SQL) is approved, use the appropriate tools (`execute_sql`, `visualize_data`, `explain_sql_query`) to perform the actions.
    * Execute potentially modifying SQL queries one by one if part of a multi-step approved plan, reporting the outcome of each.
6.  **Report Results & Explain Visualizations:** Clearly present the results of your actions.
    * For `execute_sql` (non-SELECT): Report success and rows affected, or the error.
    * **For `visualize_data`:** Report the file path to the saved plot. **Then, provide a brief textual summary interpreting what the chart illustrates based on the data queried, the chart type, and the plotted columns** (e.g., "This bar chart shows the total sales per category, with Category A having the highest sales." or "This scatter plot visualizes the relationship between X and Y, showing a positive correlation.").
    * For `explain_sql_query`: Present the query plan analysis.
    * For informational tools: Present the fetched information clearly.
7.  **Chart Analysis & Recommendation**
    * If the user provides an image of a chart along with their request (you will receive the image data directly in the input):
        * **Analyze the Image:** Carefully examine the chart image provided.
        * **Describe:** Identify the chart type (bar, line, pie, scatter, etc.), the axes (if applicable), title, legend, and the main trends, patterns, or comparisons shown.
        * **Explain & Provide Insights:** Explain what the chart communicates. Based primarily on the visual information, provide potential insights, key takeaways, or outliers visible in the chart.
        * **Critique Effectiveness:** Evaluate how effectively the chart conveys its message. Consider clarity, appropriate chart type choice for the data shown, potential clutter, misleading elements, or areas for improvement (e.g., better labels, different scales).
        * **Recommend Alternatives:** Suggest alternative chart types or improvements (e.g., "A line chart might show the trend over time more clearly," "Consider using a log scale for the Y-axis," "Reducing the number of categories might make this pie chart more readable"). Explain *why* your suggestions would be better.
        * **Connect to Data (If Possible):** If you have context about the underlying data (e.g., you generated this chart previously using `visualize_data`), try to connect your visual analysis with the known data source for more robust insights and recommendations. Otherwise, base your analysis purely on the image.

**Important Guidelines:**

* **Prioritize Safety:** Never execute data modification queries (`UPDATE`, `INSERT`, `DELETE`, `ALTER`, `DROP`) without explicit user confirmation of the exact SQL. Be extra cautious with queries lacking specific `WHERE` clauses. Avoid generating potentially destructive queries like `DROP TABLE` unless specifically and repeatedly confirmed by the user.
* **Use Tools Intelligently:** Don't guess table structures or relationships; use `get_table_schema` and `find_related_tables`. Use `explain_sql_query` to analyze performance *before* suggesting complex index changes.
* **Be Precise:** Generate correct SQL syntax for PostgreSQL. Use quotes (`"`) around identifiers if needed.
* **Handle Errors:** If a tool returns an error, report it to the user and try to adapt your plan.
* **Stay Focused:** Only perform actions directly related to the user's request and the database interaction task.

Your goal is to be a reliable, knowledgeable, and safe assistant for database tasks. Always think step-by-step, gather information, confirm actions, and communicate clearly, including explaining the visualizations you create. """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the OpenAI Tools agent - this agent type is designed to work well with structured tools
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True # More robust parsing error handling
    # You might add max_iterations if needed: max_iterations=10
)

# --- INTERACTIVE CHAT LOOP ---
# ... (keep your existing chat loop)
if __name__ == "__main__":
    print(f"Database Interaction Agent Initialized.")
    print(f"Chart images should be placed in '{LOCAL_CHART_DIR}'.")
    print(f"To analyze a chart, mention its path like: analyze `local_chart_images/your_chart.png`")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input_text = input("User: ")
            if user_input_text.lower() == 'exit':
                break

            # --- Multimodal Input Processing ---
            image_path_match = re.search(r"`([^`]+\.(?:png|jpg|jpeg|gif|webp))`", user_input_text) # Simple regex for paths in backticks
            agent_input = None

            if image_path_match:
                relative_image_path = image_path_match.group(1)
                # Assume path is relative to the script or the defined LOCAL_CHART_DIR
                # Prioritize path within LOCAL_CHART_DIR if it's not absolute
                potential_path = Path(relative_image_path)
                if not potential_path.is_absolute() and not relative_image_path.startswith(str(LOCAL_CHART_DIR)):
                     image_path = LOCAL_CHART_DIR / potential_path
                else:
                     image_path = potential_path # Use the path as is if absolute or already includes dir

                print(f"Attempting to load image: {image_path}")
                base64_image, mime_type = encode_image_to_base64(image_path)

                if base64_image and mime_type:
                    # Prepare multimodal input for LangChain
                    # The structure depends on how the PromptTemplate expects {input}
                    # Assuming {input} can now handle a list of content blocks:
                    agent_input_content = [
                        {"type": "text", "text": user_input_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        }
                    ]
                    # The key ('input') must match the placeholder in your ChatPromptTemplate
                    agent_input = {"input": agent_input_content}
                    print("Image loaded successfully. Sending multimodal input to agent.")
                else:
                    print(f"Agent: Could not load or encode the image specified: {relative_image_path}. Please ensure the path is correct and the file is a valid image in '{LOCAL_CHART_DIR}'. Proceeding with text only.")
                    # Fallback to text-only if image loading fails
                    agent_input = {"input": user_input_text}
            else:
                # No image path detected, proceed with text only
                agent_input = {"input": user_input_text}

            # --- Agent Invocation ---
            # Pass the prepared input (potentially multimodal)
            # Ensure agent_input includes necessary keys like 'chat_history' if your agent/memory requires them explicitly here
            # If memory is handled implicitly by the executor, just passing 'input' might be enough.
            # Let's assume memory is handled implicitly:
            response = agent_executor.invoke(agent_input)

            print(f"Agent: {response['output']}")

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            # Optional: break or continue based on desired robustness