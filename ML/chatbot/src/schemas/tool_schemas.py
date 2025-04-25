from pydantic.v1 import BaseModel, Field # Using v1 for potential Langchain compatibility
from src.config import MAX_SAMPLE_ROWS, DISTRIBUTION_TOP_N

# Schemas for Structured Tools

class ProfileTableSchema(BaseModel):
    table_name: str = Field(description="The name of the table to profile.")

class GetSampleDataSchema(BaseModel):
    table_name: str = Field(description="The name of the table to fetch sample data from.")
    num_rows: int = Field(description=f"The number of sample rows to fetch (max {MAX_SAMPLE_ROWS}).", default=10)

class ExecuteSQLSchema(BaseModel):
    sql_query: str = Field(description="The SQL query string to execute.")

class ExplainSQLSchema(BaseModel):
    sql_query: str = Field(description="The SQL query to analyze using EXPLAIN ANALYZE.")

class VisualizeDataSchema(BaseModel):
    sql_query: str = Field(description="The SELECT query to fetch data for plotting.")
    chart_type: str = Field(description="Type of chart ('bar', 'line', 'scatter', 'pie', 'hist').")
    x_col: str = Field(description="The column name for the X-axis (or data column for hist/pie).")
    y_col: str | None = Field(description="The column name for the Y-axis (REQUIRED for 'bar', 'line', 'scatter', 'pie').", default=None)
    color_col: str | None = Field(description="Optional column for color encoding (hue).", default=None)

class GetTableSchemaInput(BaseModel):
    table_name: str = Field(description="The name of the table to get the schema for.")

class FindRelatedTablesInput(BaseModel):
    table_name: str = Field(description="The name of the table to find foreign key relationships for.")

class GetColumnValueDistributionInput(BaseModel):
    table_name: str = Field(description="The name of the table containing the column.")
    column_name: str = Field(description="The name of the column to analyze the value distribution of.")
    top_n: int = Field(description=f"For categorical columns, the maximum number of top values to return.", default=DISTRIBUTION_TOP_N)

class GetTableIndexesInput(BaseModel):
    table_name: str = Field(description="The name of the table to get index information for.")