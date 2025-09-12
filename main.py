
# main.py (Fixed connection handling)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict, Literal
import sqlite3
import os
import re
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import warnings
from prophet import Prophet
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Load environment variables
load_dotenv()

app = FastAPI(title="Northwind DB Query API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str



# Define the state for our graph
class AgentState(TypedDict):
    query: str
    intent: Literal["data_query", "forecasting", "unknown"]
    sql_query: str
    sql_result: Any
    query_execution_success: bool
    forecast_results: Any
    answer: str






# Initialize components
def initialize_components():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./northwind.db")
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
    db = SQLDatabase.from_uri(database_url)
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    
    print("="*50)
    print("DATABASE SCHEMA:")
    print("="*50)
    print("Table Names:", db.get_usable_table_names())
    print("\nDetailed Schema:")
    print(db.get_table_info())
    print("="*50)




    sql_prompt = ChatPromptTemplate.from_template("""
IMPORTANT: Assume today's date is November 20, 2023. When the user mentions relative dates like "last month", "this year", "recently", etc., calculate them relative to November 20, 2023.

Create a SQL query for {dialect} database. Available tables: {table_info}

CRITICAL INSTRUCTIONS:
- For FORECASTING questions (predict, forecast, future), DO NOT calculate averages or forecasts in SQL
- For forecasting, retrieve HISTORICAL TIME SERIES data with date and value columns
- Get exactly 6 months of historical data (from May 2023 to October 2023, since today is November 20, 2023)
- Use monthly aggregation: GROUP BY strftime('%Y-%m', date_column)  
- Return columns named 'date' (or similar) and 'value' (or similar)
- Example for sales: SELECT strftime('%Y-%m-01', OrderDate) as date, SUM(total_amount) as value FROM orders WHERE OrderDate >= '2023-05-01' AND OrderDate < '2023-11-01' GROUP BY strftime('%Y-%m', OrderDate)

Question: {input}

Return only the SQL query without any explanation.
""")

# Also update the intent classification to be more explicit
    intent_prompt = ChatPromptTemplate.from_template("""
    IMPORTANT: Assume today's date is November 20, 2023. Use this date when interpreting any relative time references like "last month", "this year", "recently", etc.

    Classify this question as either 'data_query' or 'forecasting'. 

    FORECASTING questions include: predict, forecast, future, next month(s)/year(s), projection, trend analysis
    DATA_QUERY questions include: show, list, what, how much, total, count, average (of historical data)

    Return only one word: 'data_query' or 'forecasting'

    Question: {input}
    """)
    
    answer_prompt = ChatPromptTemplate.from_template("""
    IMPORTANT: Assume today's date is November 20, 2023. When interpreting results or providing context, use November 20, 2023 as the reference date.
    
    Answer this question based on the provided data:
    Question: {question}
    SQL Query: {query}
    Query Results: {result}
    Forecast Results: {forecast}
    
    Provide a clear, helpful answer in natural language.
    """)
    
    return llm, db, execute_query_tool, intent_prompt, sql_prompt, answer_prompt

# Initialize components
try:
    llm, db, execute_query_tool, intent_prompt, sql_prompt, answer_prompt = initialize_components()
    print("Components initialized successfully")
except Exception as e:
    print(f"Error initializing components: {e}")
    llm, db, execute_query_tool, intent_prompt, sql_prompt, answer_prompt = None, None, None, None, None, None






def generate_forecast(data, periods=6, granularity="M"):
    """Forecast using Prophet with flexible granularity: M (month), Q (quarter), Y (year)"""
    print(f"DEBUG: generate_forecast called with {len(data) if data else 0} points, granularity={granularity}")
    print(f"DEBUG: Input data sample: {data[:3] if data and len(data) > 0 else 'No data'}")

    if not data or len(data) < 4:
        print("DEBUG: Insufficient data for forecasting")
        return {"error": "Insufficient data for forecasting. Need at least 4 points."}

    try:
        # Create DataFrame for Prophet
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').dropna()
        
        print(f"DEBUG: Prophet input DataFrame shape: {df.shape}")
        print(f"DEBUG: Prophet input DataFrame head: {df.head()}")

        # Rename columns for Prophet (ds = datestamp, y = value)
        prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
        
        print(f"DEBUG: Prophet DataFrame after rename: {prophet_df.head()}")
        print(f"DEBUG: Prophet DataFrame dtypes: {prophet_df.dtypes}")

        # Initialize Prophet model with appropriate settings
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True if len(prophet_df) >= 24 else False,  # Only if we have 2+ years
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,  # More conservative changepoints
            seasonality_prior_scale=10.0,  # Allow seasonality to be learned
            interval_width=0.95,  # 95% confidence intervals
        )

        # Add custom seasonality based on granularity (simplified for 6 months of data)
        if granularity == "M" and len(prophet_df) >= 6:
            # Light monthly seasonality since we only have 6 months
            model.add_seasonality(name="monthly", period=30.5, fourier_order=2)
        elif granularity == "Q" and len(prophet_df) >= 4:
            model.add_seasonality(name="quarterly", period=91.5, fourier_order=2)

        print("DEBUG: Fitting Prophet model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)

        print("DEBUG: Model fitted successfully")

        # Choose appropriate frequency for future predictions
        freq_map = {"M": "MS", "Q": "QS", "Y": "YS"}  # Start of period
        freq = freq_map.get(granularity, "MS")

        print(f"DEBUG: Creating future dataframe with freq={freq}, periods={periods}")
        
        # Create future dataframe
        future_df = model.make_future_dataframe(periods=periods, freq=freq)
        print(f"DEBUG: Future dataframe shape: {future_df.shape}")
        print(f"DEBUG: Future dataframe tail: {future_df.tail()}")

        # Make predictions
        print("DEBUG: Making predictions...")
        forecast = model.predict(future_df)
        print(f"DEBUG: Forecast shape: {forecast.shape}")
        print(f"DEBUG: Forecast columns: {forecast.columns.tolist()}")

        # Extract forecast results (only future periods)
        forecast_results = []
        future_forecast = forecast.tail(periods)
        
        print(f"DEBUG: Future forecast shape: {future_forecast.shape}")
        
        for _, row in future_forecast.iterrows():
            forecast_results.append({
                "date": row['ds'].strftime("%Y-%m-%d"),
                "value": float(max(0, row['yhat'])),  # Ensure non-negative
                "lower_bound": float(max(0, row['yhat_lower'])),
                "upper_bound": float(max(0, row['yhat_upper'])),
                "type": "forecast"
            })

        # Add historical data for reference
        historical_results = []
        for _, row in prophet_df.iterrows():
            historical_results.append({
                "date": row['ds'].strftime("%Y-%m-%d"), 
                "value": float(row['y']), 
                "type": "historical"
            })

        print(f"DEBUG: Generated {len(forecast_results)} forecast points")
        print(f"DEBUG: Generated {len(historical_results)} historical points")
        print(f"DEBUG: Sample forecast result: {forecast_results[0] if forecast_results else 'No forecast'}")

        return historical_results + forecast_results

    except Exception as e:
        import traceback
        print(f"DEBUG: Forecasting error: {str(e)}")
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return {"error": f"Forecasting error: {str(e)}"}




# Node functions
def classify_intent(state: AgentState) -> AgentState:
    if not llm:
        return {"intent": "unknown"}
    
    try:
        response = llm.invoke(intent_prompt.format(input=state["query"]))
        intent = response.content.strip().lower()
        return {"intent": intent if intent in ["data_query", "forecasting"] else "unknown"}
    except Exception as e:
        print(f"Intent classification error: {e}")
        return {"intent": "unknown"}

def generate_sql_query(state: AgentState) -> AgentState:
    if not llm or not db:
        return {"sql_query": "", "query_execution_success": False}
    
    try:
        table_info = db.get_table_info()
        dialect = db.dialect
        
        response = llm.invoke(sql_prompt.format(
            input=state["query"],
            dialect=dialect,
            table_info=table_info
        ))
        
        sql_query = clean_sql_query(response.content)
        return {"sql_query": sql_query, "query_execution_success": True}
    except Exception as e:
        print(f"SQL generation error: {e}")
        return {"sql_query": f"-- Error: {str(e)}", "query_execution_success": False}





def execute_query(state: AgentState) -> AgentState:
    if not execute_query_tool or not state.get("sql_query"):
        return {"sql_result": "No query to execute", "query_execution_success": False}
    
    try:
        sql_result = execute_query_tool.invoke(state["sql_query"])
        print(f"DEBUG: SQL result type: {type(sql_result)}")
        print(f"DEBUG: SQL result content: {sql_result}")
        
        # Let's try a different approach - use the database directly
        # to get the results in a more predictable format
        try:
            # Use the database connection directly
            with sqlite3.connect('./northwind.db') as conn:
                cursor = conn.cursor()
                cursor.execute(state["sql_query"])
                
                # Get column names
                columns = [description[0] for description in cursor.description]
                
                # Get all rows
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                result_list = []
                for row in rows:
                    result_list.append(dict(zip(columns, row)))
                
                print(f"DEBUG: Direct query results: {result_list}")
                return {"sql_result": result_list, "query_execution_success": True}
                
        except Exception as db_error:
            print(f"Direct query error: {db_error}")
            # Fall back to the original tool result
            return {"sql_result": sql_result, "query_execution_success": True}
            
    except Exception as e:
        print(f"Query execution error: {e}")
        return {"sql_result": f"Error: {str(e)}", "query_execution_success": False}

def process_forecasting(state: AgentState) -> AgentState:
    if not state.get("query_execution_success", False):
        return {"forecast_results": {"error": "Query execution failed"}}
    
    try:
        sql_result = state["sql_result"]
        data = parse_sql_result(sql_result)
        
        if not data or ("error" in data if isinstance(data, dict) else False):
            return {"forecast_results": {"error": "No suitable data found for forecasting"}}
        
        forecast_results = generate_forecast(data)
        return {"forecast_results": forecast_results}
    except Exception as e:
        print(f"Forecasting error: {e}")
        return {"forecast_results": {"error": f"Forecasting failed: {str(e)}"}}

def generate_final_answer(state: AgentState) -> AgentState:
    if not llm:
        return {"answer": "System error: LLM not available"}
    
    try:
        response = llm.invoke(answer_prompt.format(
            question=state["query"],
            query=state.get("sql_query", ""),
            result=state.get("sql_result", ""),
            forecast=state.get("forecast_results", "")
        ))
        return {"answer": response.content}
    except Exception as e:
        print(f"Answer generation error: {e}")
        return {"answer": f"Error generating answer: {str(e)}"}

# Graph workflow
def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("execute_sql", execute_query)
    workflow.add_node("process_forecast", process_forecasting)
    workflow.add_node("generate_answer", generate_final_answer)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Define routing
    def route_by_intent(state):
        return state.get("intent", "unknown")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "data_query": "generate_sql",
            "forecasting": "generate_sql",
            "unknown": "generate_answer"
        }
    )
    
    workflow.add_edge("generate_sql", "execute_sql")
    
    def route_after_execution(state):
        if not state.get("query_execution_success", False):
            return "generate_answer"
        return "process_forecast" if state.get("intent") == "forecasting" else "generate_answer"
    
    workflow.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "process_forecast": "process_forecast",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_edge("process_forecast", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# Helper functions
def clean_sql_query(sql_query: str) -> str:
    if not sql_query:
        return ""
    sql_query = re.sub(r'```sql\s*', '', sql_query)
    sql_query = re.sub(r'```', '', sql_query)
    lines = sql_query.strip().split('\n')
    sql_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('#', '--'))]
    return ' '.join(sql_lines).strip()







def parse_sql_result(result, granularity="M"):
    """Parse SQL result and prepare time series data for Prophet forecasting"""
    try:
        data = None
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    data = parsed
            except json.JSONDecodeError:
                pass
        elif isinstance(result, list):
            data = result

        if not data:
            return {"error": "No data to process"}

        df = pd.DataFrame(data)
        print(f"DEBUG: Original DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
        print(f"DEBUG: DataFrame head: {df.head()}")

        # Detect columns - be more flexible with date detection
        date_col, value_col = None, None
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['date', 'month', 'time', 'year', 'period', 'orderdate']):
                date_col = col
            elif any(term in col_lower for term in ['value', 'amount', 'total', 'sales', 'revenue', 'price', 'sum']):
                value_col = col

        print(f"DEBUG: Detected date_col: {date_col}, value_col: {value_col}")

        if not date_col or not value_col:
            return {"error": f"Missing required columns. Found: {df.columns.tolist()}"}

        # Convert and clean
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Remove rows with null dates or values
        before_clean = len(df)
        df = df.dropna(subset=[date_col, value_col])
        after_clean = len(df)
        print(f"DEBUG: Cleaned data: {before_clean} -> {after_clean} rows")
        
        if df.empty:
            return {"error": "No valid data after cleaning"}

        # Sort by date
        df = df.sort_values(date_col)
        
        # Group by the specified time period and sum values
        if granularity == "Q":
            # Group by quarter
            df_grouped = df.groupby(df[date_col].dt.to_period("Q"))[value_col].sum().reset_index()
            df_grouped[date_col] = df_grouped[date_col].dt.to_timestamp()
        elif granularity == "Y":
            # Group by year
            df_grouped = df.groupby(df[date_col].dt.to_period("Y"))[value_col].sum().reset_index()
            df_grouped[date_col] = df_grouped[date_col].dt.to_timestamp()
        else:  # Default monthly
            # Group by month
            df_grouped = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum().reset_index()
            df_grouped[date_col] = df_grouped[date_col].dt.to_timestamp()

        print(f"DEBUG: Grouped DataFrame shape: {df_grouped.shape}")
        print(f"DEBUG: Grouped DataFrame head: {df_grouped.head()}")
        
        # Convert to the format expected by Prophet
        result_data = []
        for _, row in df_grouped.iterrows():
            result_data.append({
                "date": row[date_col].strftime("%Y-%m-%d"), 
                "value": float(row[value_col])
            })
        
        print(f"DEBUG: Final result data length: {len(result_data)}")
        print(f"DEBUG: Sample result data: {result_data[:3] if result_data else 'No data'}")
        
        return result_data

    except Exception as e:
        import traceback
        print(f"DEBUG: Parse error: {str(e)}")
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return {"error": f"Parse error: {str(e)}"}



# Initialize workflow
try:
    workflow = create_workflow()
    print("Workflow initialized successfully")
except Exception as e:
    print(f"Error initializing workflow: {e}")
    workflow = None

# API endpoints
@app.get("/")
async def read_root():
    return {"message": "Northwind DB Query API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "workflow_initialized": workflow is not None,
        "database_connected": db is not None,
        "llm_initialized": llm is not None
    }

# Helper function to parse and simplify schema
def get_simplified_schema():
    """Extract simplified table structure from database"""
    if not db:
        return {"error": "Database not available"}
    
    try:
        tables = db.get_usable_table_names()
        table_info = db.get_table_info()
        
        # Parse the schema to extract table structures
        simplified_tables = {}
        
        for table in tables:
            # Extract CREATE TABLE statement for each table
            table_pattern = rf'CREATE TABLE ["\']?{re.escape(table)}["\']?\s*\((.*?)\)'
            match = re.search(table_pattern, table_info, re.DOTALL | re.IGNORECASE)
            
            if match:
                columns_text = match.group(1)
                columns = []
                
                # Parse column definitions
                for line in columns_text.split('\n'):
                    line = line.strip().rstrip(',')
                    if line and not line.startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CHECK', 'UNIQUE')):
                        # Extract column name and type
                        parts = line.split()
                        if len(parts) >= 2:
                            col_name = parts[0].strip('"')
                            col_type = parts[1]
                            columns.append({"name": col_name, "type": col_type})
                
                simplified_tables[table] = {
                    "columns": columns,
                    "column_count": len(columns)
                }
        
        return {
            "tables": tables,
            "table_details": simplified_tables,
            "total_tables": len(tables)
        }
        
    except Exception as e:
        return {"error": f"Failed to parse schema: {str(e)}"}
    

def parse_sql_result_for_frontend(sql_result):
    """Parse SQL result for frontend consumption"""
    if not sql_result or sql_result == "No query to execute" or isinstance(sql_result, str) and "Error" in sql_result:
        return []
    
    try:
        # If it's already a list, return it
        if isinstance(sql_result, list):
            return sql_result
        
        # If it's a string, try to parse as JSON
        if isinstance(sql_result, str):
            # Try direct JSON parsing first
            try:
                return json.loads(sql_result)
            except json.JSONDecodeError:
                # Try to extract JSON from the string
                if '[' in sql_result and ']' in sql_result:
                    start_idx = sql_result.find('[')
                    end_idx = sql_result.rfind(']') + 1
                    json_str = sql_result[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # If it's a simple string result, convert to a list with one item
                    return [{"result": sql_result}]
        
        return []
    except Exception as e:
        print(f"Error parsing SQL result: {e}")
        return []



@app.post("/query")
async def process_query(request: QueryRequest):
    if not workflow:
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    try:
        result = workflow.invoke({"query": request.query})
        
        # Get simplified schema information
        schema_info = get_simplified_schema()
        
        # Extract and parse SQL results
        sql_result = result.get("sql_result", "")
        parsed_results = parse_sql_result_for_frontend(sql_result)
        
        print(f"DEBUG: Parsed results: {parsed_results}")
        print(f"DEBUG: Results type: {type(parsed_results)}")
        print(f"DEBUG: Results length: {len(parsed_results)}")
        
        return {
            "answer": result.get("answer", "No answer generated"),
            "sqlQuery": result.get("sql_query", ""),
            "results": parsed_results,
            "forecastResults": result.get("forecast_results", []),
            "schema": schema_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/debug-forecast")
async def debug_forecast(request: QueryRequest):
    """Debug endpoint to test forecasting without LLM"""
    if not workflow:
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    try:
        # First, let's test the SQL generation and execution
        state = {"query": request.query, "intent": "forecasting"}
        
        # Generate SQL
        sql_result = generate_sql_query(state)
        print(f"DEBUG API: SQL Query: {sql_result.get('sql_query')}")
        
        # Execute SQL
        state.update(sql_result)
        execution_result = execute_query(state)
        print(f"DEBUG API: SQL Result: {execution_result.get('sql_result')}")
        
        # Parse the results
        sql_data = execution_result.get('sql_result')
        parsed_data = parse_sql_result(sql_data)
        print(f"DEBUG API: Parsed Data: {parsed_data}")
        
        # Test forecasting
        if isinstance(parsed_data, list) and len(parsed_data) > 0:
            forecast_result = generate_forecast(parsed_data)
            print(f"DEBUG API: Forecast Result: {forecast_result}")
            
            return {
                "sql_query": sql_result.get('sql_query'),
                "sql_result": sql_data,
                "parsed_data": parsed_data,
                "forecast_result": forecast_result,
                "data_points": len(parsed_data) if isinstance(parsed_data, list) else 0
            }
        else:
            return {
                "error": "No valid data for forecasting",
                "sql_query": sql_result.get('sql_query'),
                "sql_result": sql_data,
                "parsed_data": parsed_data
            }
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
