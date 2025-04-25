import json
from sqlalchemy import inspect, text, func
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError, OperationalError
from src.database.connection import engine # Import the shared engine
from src.utils.helpers import table_exists # Import utility
from src.config import DISTRIBUTION_TOP_N # Import default N

# Note: All functions now implicitly use the 'engine' imported above

def list_tables(*args, **kwargs) -> str:
    """Return a list of table names in the connected database as a JSON string."""
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return json.dumps(tables)
    except Exception as e:
        return f"Error listing tables: {e}"

def get_table_schema(table_name: str) -> str:
    """
    Retrieves detailed schema information for a specific table.
    Returns a description of the schema as a JSON string or an error message.
    """
    print(f"--- Getting schema for table: {table_name} ---")
    try:
        inspector = inspect(engine)

        if not table_exists(inspector, table_name):
            return f"Error: Table '{table_name}' not found in the database."

        schema_info = {}
        schema_info['table_name'] = table_name
        # Use safe methods to get info, handle potential None returns
        schema_info['columns'] = [
            {k: str(v) if k == 'type' else v for k, v in col.items()}
            for col in inspector.get_columns(table_name)
        ]
        schema_info['primary_key'] = inspector.get_pk_constraint(table_name)
        schema_info['foreign_keys'] = inspector.get_foreign_keys(table_name)
        schema_info['indexes'] = inspector.get_indexes(table_name)
        # try: # Table comment might not be supported/exist
        #     schema_info['comment'] = inspector.get_table_comment(table_name)
        # except Exception:
        #     schema_info['comment'] = None

        # Use json.dumps for clean, readable dictionary output
        return json.dumps(schema_info, indent=2, default=str) # default=str for safety

    except NoSuchTableError:
        return f"Error: Table '{table_name}' not found."
    except SQLAlchemyError as db_err:
        return f"Database error while fetching schema for {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error occurred while getting schema for {table_name}: {e}"


def find_related_tables(table_name: str) -> str:
    """
    Finds tables related via foreign keys (outgoing and incoming).
    Returns relationships as a JSON string or an error message.
    """
    print(f"--- Finding relationships for table: {table_name} ---")
    try:
        inspector = inspect(engine)
        if not table_exists(inspector, table_name):
             return f"Error: Table '{table_name}' not found in the database."

        relationships = {"table": table_name, "references": [], "referenced_by": []}
        # Outgoing FKs
        outgoing_fks = inspector.get_foreign_keys(table_name)
        for fk in outgoing_fks:
             relationships["references"].append(fk) # FK dict is usually serializable

        # Incoming FKs (can be slow)
        print(f"Checking incoming references to {table_name}...")
        all_tables = inspector.get_table_names()
        all_schemas = inspector.get_schema_names() # Check all schemas if necessary

        for schema in all_schemas:
            # Skip system schemas if desired (e.g., 'information_schema', 'pg_catalog')
            if schema.startswith('pg_') or schema == 'information_schema':
                continue
            tables_in_schema = inspector.get_table_names(schema=schema)
            for other_table in tables_in_schema:
                 if schema == inspector.default_schema_name and other_table == table_name:
                     continue # Skip self-reference check within default schema
                 # Proper identification might need schema.table format
                 full_other_table_name = f"{schema}.{other_table}" if schema != inspector.default_schema_name else other_table

                 try:
                     incoming_fks = inspector.get_foreign_keys(other_table, schema=schema)
                     for fk in incoming_fks:
                         # Check if referred table matches (consider schema)
                         referred_schema = fk.get('referred_schema') or inspector.default_schema_name
                         if fk['referred_table'] == table_name and referred_schema == inspector.default_schema_name: # Adjust if target table could be in another schema
                              fk_info = fk.copy()
                              fk_info['referencing_table'] = full_other_table_name # Add qualified name
                              relationships["referenced_by"].append(fk_info)
                 except Exception as inner_e:
                    print(f"Warning: Could not check FKs for table '{full_other_table_name}': {inner_e}")

        return json.dumps(relationships, indent=2, default=str)

    except NoSuchTableError:
        return f"Error: Table '{table_name}' not found."
    except SQLAlchemyError as db_err:
        return f"Database error finding relationships for {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error finding relationships for {table_name}: {e}"

def get_column_value_distribution(table_name: str, column_name: str, top_n: int = DISTRIBUTION_TOP_N) -> str:
    """
    Analyzes value distribution for a column (numeric stats or categorical counts).
    Returns analysis as a JSON string or an error message.
    """
    print(f"--- Analyzing distribution for {table_name}.{column_name} ---")
    try:
        inspector = inspect(engine)
        if not table_exists(inspector, table_name):
            return f"Error: Table '{table_name}' not found."

        columns = inspector.get_columns(table_name)
        col_info = next((col for col in columns if col['name'] == column_name), None)
        if not col_info:
            return f"Error: Column '{column_name}' not found in table '{table_name}'."

        col_type = str(col_info['type']).upper()
        distribution = {"table": table_name, "column": column_name, "type": col_type}
        # Quote identifiers safely - PostgreSQL uses double quotes
        quoted_table = f'"{table_name}"' # Basic quoting, consider schema if needed
        quoted_col = f'"{column_name}"'

        with engine.connect() as conn:
            # --- Numeric Analysis ---
            if any(t in col_type for t in ['INT', 'FLOAT', 'DECIMAL', 'NUMERIC', 'DOUBLE', 'REAL', 'BIGINT', 'SMALLINT']):
                distribution['analysis_type'] = 'numeric_percentiles'
                # Using percentile_cont for PostgreSQL. Ensure column cast if needed.
                # Note: array[0, 0.25, 0.5, 0.75, 1] gives min, quartiles, max
                percentile_sql = text(f"""
                    SELECT
                        COUNT(*) as total_count,
                        COUNT({quoted_col}) as non_null_count,
                        AVG({quoted_col}::numeric) as mean,
                        STDDEV_POP({quoted_col}::numeric) as stddev,
                        percentile_cont(array[0, 0.25, 0.5, 0.75, 1]) WITHIN GROUP (ORDER BY {quoted_col}) as percentiles
                    FROM {quoted_table}
                """)
                try:
                    result = conn.execute(percentile_sql).fetchone()
                    if result and result._mapping.get('percentiles'):
                        p_list = result._mapping['percentiles']
                        distribution.update({
                            "total_count": result._mapping.get('total_count'),
                            "non_null_count": result._mapping.get('non_null_count'),
                            "null_count": (result._mapping.get('total_count') or 0) - (result._mapping.get('non_null_count') or 0),
                            "min": p_list[0],
                            "percentile_25": p_list[1],
                            "percentile_50_median": p_list[2],
                            "percentile_75": p_list[3],
                            "max": p_list[4],
                            "mean": float(result._mapping['mean']) if result._mapping.get('mean') is not None else None,
                            "stddev": float(result._mapping['stddev']) if result._mapping.get('stddev') is not None else None,
                        })
                    else: # Handle empty table or all nulls
                         count_sql = text(f"SELECT COUNT(*) as total_count, COUNT({quoted_col}) as non_null_count FROM {quoted_table}")
                         count_result = conn.execute(count_sql).fetchone()
                         distribution.update({
                            "total_count": count_result._mapping.get('total_count'),
                            "non_null_count": count_result._mapping.get('non_null_count'),
                            "null_count": (count_result._mapping.get('total_count') or 0) - (count_result._mapping.get('non_null_count') or 0),
                            "min": None, "percentile_25": None, "percentile_50_median": None, "percentile_75": None, "max": None, "mean": None, "stddev": None,
                            "message": "Not enough non-null data for numeric stats."
                         })

                except OperationalError as op_err:
                    distribution['error'] = f"Could not calculate numeric distribution (DB function might be unsupported or query failed): {op_err}"
                    # Add fallback if needed, e.g., just min/max/avg
                except Exception as num_err:
                     distribution['error'] = f"Error during numeric analysis: {num_err}"

            # --- Categorical Analysis ---
            elif any(t in col_type for t in ['CHAR', 'TEXT', 'VARCHAR', 'STRING', 'BOOLEAN', 'ENUM', 'UUID']): # Add other relevant types
                distribution['analysis_type'] = 'categorical_counts'
                counts_sql = text(f"""
                    SELECT {quoted_col}, COUNT(*) as count
                    FROM {quoted_table}
                    WHERE {quoted_col} IS NOT NULL
                    GROUP BY {quoted_col}
                    ORDER BY count DESC
                    LIMIT :limit
                """)
                null_count_sql = text(f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_col} IS NULL")
                distinct_count_sql = text(f"SELECT COUNT(DISTINCT {quoted_col}) FROM {quoted_table}")
                total_count_sql = text(f"SELECT COUNT(*) FROM {quoted_table}")

                try:
                    top_values = conn.execute(counts_sql, {'limit': top_n}).fetchall()
                    null_count = conn.execute(null_count_sql).scalar_one_or_none()
                    distinct_count = conn.execute(distinct_count_sql).scalar_one_or_none()
                    total_count = conn.execute(total_count_sql).scalar_one_or_none()

                    distribution['total_count'] = total_count
                    distribution['distinct_count'] = distinct_count
                    distribution['null_count'] = null_count
                    distribution['non_null_count'] = (total_count or 0) - (null_count or 0)
                    distribution['top_values'] = [dict(row._mapping) for row in top_values]

                except Exception as cat_err:
                     distribution['error'] = f"Error during categorical analysis: {cat_err}"

            # --- Other Types ---
            else:
                distribution['analysis_type'] = 'other'
                distribution['message'] = f"Basic distribution analysis implemented. Type: '{col_type}'. Fetching null count."
                try:
                    null_count_sql = text(f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_col} IS NULL")
                    total_count_sql = text(f"SELECT COUNT(*) FROM {quoted_table}")
                    null_count = conn.execute(null_count_sql).scalar_one_or_none()
                    total_count = conn.execute(total_count_sql).scalar_one_or_none()
                    distribution['total_count'] = total_count
                    distribution['null_count'] = null_count
                    distribution['non_null_count'] = (total_count or 0) - (null_count or 0)
                except Exception as other_err:
                    distribution['error'] = f"Error fetching counts for type {col_type}: {other_err}"

        return json.dumps(distribution, indent=2, default=str) # Use default=str for non-serializable types like Decimal

    except NoSuchTableError:
        return f"Error: Table '{table_name}' not found."
    except KeyError:
        return f"Error: Column '{column_name}' not found in table '{table_name}' (post-check)."
    except SQLAlchemyError as db_err:
        return f"Database error analyzing distribution for {table_name}.{column_name}: {db_err}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Unexpected error analyzing distribution for {table_name}.{column_name}: {e}"


def get_table_indexes(table_name: str) -> str:
    """
    Retrieves index information for a specific table.
    Returns index list as a JSON string or an error message.
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
        return f"Database error fetching indexes for {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error getting indexes for {table_name}: {e}"


def profile_table(table_name: str) -> str:
    """
    Profiles a table: row count, column stats (nulls, distinct, numeric, string).
    Returns profile as a JSON string or an error message.
    """
    print(f"--- Profiling table: {table_name} ---")
    try:
        inspector = inspect(engine)
        if not table_exists(inspector, table_name):
            return f"Error: Table '{table_name}' not found."

        cols = inspector.get_columns(table_name)
        profile = {"table": table_name, "columns": {}}
        quoted_table = f'"{table_name}"' # Basic quoting

        with engine.connect() as conn:
            total_rows_result = conn.execute(text(f'SELECT COUNT(*) FROM {quoted_table}'))
            total_rows = total_rows_result.scalar_one_or_none() or 0 # Handle empty table
            profile['total_rows'] = total_rows

            for col in cols:
                name = col["name"]
                dtype = col["type"]
                stats = {'type': str(dtype)} # Store type info
                quoted_name = f'"{name}"' # Basic quoting

                # Null count/rate
                null_q = f'SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_name} IS NULL'
                null_count = conn.execute(text(null_q)).scalar_one_or_none() or 0
                stats['null_count'] = null_count
                stats['null_rate'] = (null_count / total_rows) if total_rows > 0 else 0

                # Distinct count
                try:
                    distinct_q = f'SELECT COUNT(DISTINCT {quoted_name}) FROM {quoted_table}'
                    stats['distinct_count'] = conn.execute(text(distinct_q)).scalar_one_or_none()
                except Exception as distinct_e:
                    print(f"Warning: Could not get distinct count for {table_name}.{name}: {distinct_e}")
                    stats['distinct_count'] = "Error" # Indicate error

                # Numeric stats
                dtype_str = str(dtype).upper()
                if any(t in dtype_str for t in ['INT', 'FLOAT', 'DECIMAL', 'NUMERIC', 'DOUBLE', 'REAL']):
                    if total_rows > null_count: # Only calculate if non-null values exist
                        try:
                            # Cast to numeric for consistent AVG/STDDEV across types
                            num_q = f"""
                                SELECT
                                    MIN({quoted_name}),
                                    MAX({quoted_name}),
                                    AVG({quoted_name}::numeric),
                                    STDDEV_POP({quoted_name}::numeric)
                                FROM {quoted_table} WHERE {quoted_name} IS NOT NULL
                            """
                            mn, mx, avg, sd = conn.execute(text(num_q)).fetchone()
                            stats.update({
                                'min': mn, 'max': mx,
                                'mean': float(avg) if avg is not None else None,
                                'stddev': float(sd) if sd is not None else None
                            })
                        except Exception as num_e:
                            print(f"Warning: Could not calculate numeric stats for {table_name}.{name}: {num_e}")
                            stats.update({ 'min': None, 'max': None, 'mean': None, 'stddev': "Error" })
                    else:
                         stats.update({ 'min': None, 'max': None, 'mean': None, 'stddev': None })


                # String stats (Avg Length)
                elif any(t in dtype_str for t in ['CHAR', 'TEXT', 'VARCHAR', 'STRING']):
                     if total_rows > null_count:
                        try:
                            len_q = f"SELECT AVG(LENGTH({quoted_name})) FROM {quoted_table} WHERE {quoted_name} IS NOT NULL"
                            avg_len_result = conn.execute(text(len_q)).scalar_one_or_none()
                            stats['avg_length'] = float(avg_len_result) if avg_len_result is not None else None
                        except Exception as str_e:
                            print(f"Warning: Could not calculate avg length for {table_name}.{name}: {str_e}")
                            stats['avg_length'] = "Error"
                     else:
                         stats['avg_length'] = None


                profile['columns'][name] = stats

        return json.dumps(profile, indent=2, default=str) # Return as JSON string

    except NoSuchTableError:
        return f"Error: Table '{table_name}' not found."
    except SQLAlchemyError as db_err:
        return f"Database error profiling table {table_name}: {db_err}"
    except Exception as e:
        return f"An unexpected error occurred profiling table {table_name}: {e}"