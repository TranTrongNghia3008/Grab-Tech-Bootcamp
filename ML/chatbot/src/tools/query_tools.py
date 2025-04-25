import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from src.database.connection import engine # Import shared engine
from src.config import MAX_SAMPLE_ROWS

def get_sample_data(table_name: str, num_rows: int = 10) -> str:
    """
    Fetches a sample of rows from the specified table.
    Returns data as a string (DataFrame repr) or an error message.
    """
    print(f"--- Getting sample data for table: {table_name} (rows: {num_rows}) ---")
    try:
        # Validate and sanitize num_rows
        num_rows = max(1, min(num_rows, MAX_SAMPLE_ROWS))
        # Basic quoting for table name - consider schema if needed
        quoted_table = f'"{table_name}"'
        query = text(f'SELECT * FROM {quoted_table} LIMIT :limit')

        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={'limit': num_rows})
        # Return DataFrame's string representation, handle potential large output if needed
        return df.to_string()

    except (SQLAlchemyError, ProgrammingError) as db_err: # Catch DB errors like table not found
        return f"Error fetching sample data from {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error occurred fetching sample data from {table_name}: {e}"


def execute_sql(sql_query: str) -> str:
    """
    Executes a given SQL query (SELECT, INSERT, UPDATE, DELETE, etc.).
    Returns confirmation/rows affected or an error message.
    USE WITH EXTREME CAUTION.
    """
    print(f"--- Attempting to execute SQL ---")
    print(f"Query: {sql_query}")
    print("-" * 30)
    # Basic check for common risky operations without WHERE - can be improved
    query_upper = sql_query.strip().upper()
    if (query_upper.startswith("UPDATE ") or query_upper.startswith("DELETE FROM ")) and "WHERE" not in query_upper:
         print("WARNING: Executing UPDATE/DELETE without a WHERE clause. This is highly risky.")
         # Maybe add an extra confirmation step here in a real application

    try:
        with engine.connect() as conn:
            # Use transaction for non-SELECT or multi-statement queries
            # For single statements, autocommit might be default depending on driver/DBAPI
            with conn.begin() as transaction:
                try:
                    is_select = query_upper.startswith("SELECT")
                    result_proxy = conn.execute(text(sql_query))

                    if is_select:
                        # For SELECT, we don't fetch data here, just confirm execution
                        # Fetching a single row to ensure query validity might be useful
                        # result_proxy.fetchone() # Uncomment to check validity, consumes first row
                        row_count = -1 # Indicate SELECT doesn't return affected rows count directly
                        message = "SELECT query executed successfully. Use 'get_sample_data' or 'visualize_data' to see results."
                    else:
                        # For INSERT, UPDATE, DELETE, rowcount is often available
                        # For others (CREATE, ALTER, etc.), rowcount might be -1 or behavior varies
                        row_count = result_proxy.rowcount
                        message = f"Query executed successfully. Rows affected: {row_count}"

                    transaction.commit() # Commit if execution successful
                    return message

                except (SQLAlchemyError, ProgrammingError) as exec_err:
                    print(f"Error during query execution, rolling back transaction: {exec_err}")
                    try:
                        transaction.rollback()
                    except Exception as rb_err:
                        print(f"Error during rollback: {rb_err}")
                    # Provide informative error message
                    return f"Error executing query: {exec_err}. Transaction rolled back."
                except Exception as e: # Catch other unexpected errors during execution
                     print(f"Unexpected error during query execution, attempting rollback: {e}")
                     try:
                         transaction.rollback()
                     except Exception as rb_err:
                         print(f"Error during rollback: {rb_err}")
                     return f"Unexpected error during query execution: {e}. Transaction rolled back."

    except (SQLAlchemyError, ProgrammingError) as conn_err: # Errors connecting or starting transaction
         return f"Database connection or transaction error: {conn_err}"
    except Exception as e: # Catch-all for other setup errors
         return f"An unexpected error occurred before query execution: {e}"


def explain_sql_query(sql_query: str) -> str:
    """
    Analyzes a SQL query using EXPLAIN ANALYZE (PostgreSQL).
    Returns the query plan/stats or an error message.
    """
    print(f"--- Explaining SQL query ---")
    print(f"Query: {sql_query}")
    print("-" * 30)

    # EXPLAIN ANALYZE executes the query, so wrap in transaction and always rollback
    explain_command = f"EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) {sql_query}" # Use JSON format for easier parsing if needed later
    # Alternative: EXPLAIN ANALYZE {sql_query} for text output

    try:
        with engine.connect() as conn:
            with conn.begin() as transaction: # Use transaction to ensure rollback
                try:
                    result = conn.execute(text(explain_command))
                    # Fetchall returns list of tuples, each containing one part of the plan (or one JSON string if FORMAT JSON)
                    query_plan_rows = result.fetchall()
                    transaction.rollback() # Crucial: Always rollback after EXPLAIN ANALYZE

                    # Process the result (assuming FORMAT JSON, it returns one row with a list containing one JSON object)
                    if query_plan_rows and len(query_plan_rows[0]) > 0:
                         # Extract the JSON plan object
                         plan_json = query_plan_rows[0][0] # It's usually nested [[plan_json]]
                         # Pretty print the JSON plan
                         plan_output = json.dumps(plan_json, indent=2)
                         return f"Query Plan Analysis (JSON):\n--------------------------\n{plan_output}"
                    else: # Handle case where EXPLAIN ANALYZE returns nothing (shouldn't happen on valid query)
                         return "EXPLAIN ANALYZE executed but returned no plan data."

                    # --- Alternative for TEXT format ---
                    # plan_output = "\n".join([row[0] for row in query_plan_rows])
                    # return f"Query Plan Analysis (Text):\n------------------------\n{plan_output}"
                    # --- End Alternative ---

                except (SQLAlchemyError, ProgrammingError) as exec_err:
                     print(f"Error executing EXPLAIN ANALYZE, rolling back: {exec_err}")
                     try:
                         transaction.rollback()
                     except Exception as rb_err:
                         print(f"Error during rollback: {rb_err}")
                     return f"Error executing EXPLAIN ANALYZE: {exec_err}. Query might be invalid or unsupported."
                except Exception as e:
                     print(f"Unexpected error during EXPLAIN ANALYZE, attempting rollback: {e}")
                     try:
                         transaction.rollback()
                     except Exception as rb_err:
                         print(f"Error during rollback: {rb_err}")
                     return f"Unexpected error during EXPLAIN ANALYZE: {e}. Rolled back."

    except (SQLAlchemyError, ProgrammingError) as conn_err:
        return f"Database connection or transaction error during EXPLAIN ANALYZE: {conn_err}"
    except Exception as e:
        return f"An unexpected error occurred before EXPLAIN ANALYZE: {e}"