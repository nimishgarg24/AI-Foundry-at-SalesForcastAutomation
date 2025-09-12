
// App.js (updated for LangGraph with forecasting support)
import React, { useState, useEffect } from 'react';
import ChartVisualization from './ChartVisualization';

import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [sql, setSql] = useState('');
  const [answer, setAnswer] = useState('');
  const [results, setResults] = useState([]);
  const [forecastResults, setForecastResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showSql, setShowSql] = useState(false);
  const [chartType, setChartType] = useState('bar');

// Add this function to handle chart type change
const handleChartTypeChange = (type) => {
  setChartType(type);
};


  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setError('');
    setAnswer('');
    setResults([]);
    setForecastResults([]);
    setSql('');
    
    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to process query');
      }
      
      setSql(data.sqlQuery);
      setAnswer(data.answer);
      setResults(data.results || []);
      setForecastResults(data.forecastResults || []);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const sampleQueries = [
    "Show me the top 5 customers by total orders",
    "Which products are most popular?",
    "What are the total sales by category?",
    "Show me customers from Germany",
    "Which employee has the most sales?",
    "Forecast sales for the next 6 months",
    "Predict product demand for next quarter",
    "What will be our revenue next year?",
    "Forecast order trends for the upcoming year"
  ];


  const [schema, setSchema] = useState(null);

useEffect(() => {
  fetchSchema();
}, []);

const fetchSchema = async () => {
  try {
    const response = await fetch('http://localhost:8000/schema');
    const schemaData = await response.json();
    setSchema(schemaData);
    console.log('Database schema:', schemaData);
  } catch (error) {
    console.error('Error fetching schema:', error);
  }
};


  // Check if we have forecast results with historical data
  const hasForecastData = forecastResults && 
                         Array.isArray(forecastResults) && 
                         forecastResults.length > 0 &&
                         !forecastResults.error;

  // Separate historical and forecast data if available
  let historicalData = [];
  let forecastData = [];
  
  if (hasForecastData) {
    historicalData = forecastResults.filter(item => item.type === 'historical');
    forecastData = forecastResults.filter(item => item.type === 'forecast');
  }

  // Render forecast chart if forecast results are available
  const renderForecastChart = () => {
    if (!hasForecastData) return null;
    
    // Simple chart using divs (you could integrate a proper charting library here)
    const allData = [...historicalData, ...forecastData];
    const maxValue = Math.max(...allData.map(item => item.value));
    
    return (
      <div className="forecast-container">
        <h3>📈 Forecast Visualization</h3>
        <div className="chart-container">
          <div className="chart-bars">
            {allData.map((item, index) => {
              const height = maxValue > 0 ? (item.value / maxValue) * 100 : 0;
              const isForecast = item.type === 'forecast';
              
              return (
                <div key={index} className="chart-bar-container">
                  <div 
                    className={`chart-bar ${isForecast ? 'forecast-bar' : 'historical-bar'}`}
                    style={{ height: `${height}%` }}
                    title={`${item.date}: ${item.value.toFixed(2)}`}
                  >
                    <div className="bar-value">{item.value.toFixed(0)}</div>
                  </div>
                  <div className="bar-label">
                    {index % 4 === 0 ? item.date.split('-')[0] : ''}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="chart-legend">
            <div className="legend-item">
              <div className="color-box historical-color"></div>
              <span>Historical Data</span>
            </div>
            <div className="legend-item">
              <div className="color-box forecast-color"></div>
              <span>Forecast</span>
            </div>
          </div>
        </div>
        
        <div className="forecast-table">
          <h4>Detailed Forecast Data</h4>
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Value</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {forecastResults.slice(0, 10).map((item, index) => (
                <tr key={index}>
                  <td>{item.date}</td>
                  <td>{typeof item.value === 'number' ? item.value.toFixed(2) : item.value}</td>
                  <td className={`type-${item.type}`}>{item.type}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {forecastResults.length > 10 && (
            <p className="truncated-notice">
              Showing first 10 of {forecastResults.length} data points
            </p>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Northwind DB Query Assistant</h1>
        <p>Ask questions about your business data or request forecasts</p>
        
        <form onSubmit={handleQuery} className="query-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Show me the top 5 products by sales or Forecast next quarter's revenue"
            disabled={loading}
            className="query-input"
          />
          <button type="submit" disabled={loading || !query.trim()} className="submit-btn">
            {loading ? 'Analyzing...' : 'Ask'}
          </button>
        </form>

        <div className="sample-queries">
          <p>Try these sample questions:</p>
          <div className="sample-buttons">
            {sampleQueries.map((sampleQuery, index) => (
              <button
                key={index}
                onClick={() => setQuery(sampleQuery)}
                className="sample-btn"
                disabled={loading}
              >
                {sampleQuery}
              </button>
            ))}
          </div>
        </div>
        
        {error && (
          <div className="error-container">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}
        
        {answer && (
          <div className="answer-container">
            <h2>📊 Answer</h2>
            <div className="answer-text">{answer}</div>
            
            {sql && (
              <div className="sql-toggle">
                <button 
                  onClick={() => setShowSql(!showSql)}
                  className="toggle-sql-btn"
                >
                  {showSql ? 'Hide' : 'Show'} SQL Query
                </button>
                
                {showSql && (
                  <div className="sql-container">
                    <pre className="sql-code">{sql}</pre>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {results.length > 0 && (
          <ChartVisualization 
            data={results} 
            chartType={chartType}
            onChartTypeChange={handleChartTypeChange}
          />
        )}

        
        {renderForecastChart()}
        
        {results.length > 0 && (
          <div className="results-container">
            <h2>📋 Detailed Results ({results.length} rows)</h2>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    {Object.keys(results[0]).map((key) => (
                      <th key={key}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.slice(0, 50).map((row, index) => (
                    <tr key={index}>
                      {Object.values(row).map((value, i) => (
                        <td key={i}>{String(value)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {results.length > 50 && (
                <p className="truncated-notice">
                  Showing first 50 rows of {results.length} total results
                </p>
              )}
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;