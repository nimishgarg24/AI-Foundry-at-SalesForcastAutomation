// ChartVisualization.js
import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Line, Pie } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const ChartVisualization = ({ data, chartType, onChartTypeChange }) => {
  if (!data || data.length === 0) return null;

  // Prepare data for charts
  const labels = data.map(item => {
    // Extract meaningful labels from the data
    if (item.hasOwnProperty('ProductName')) return item.ProductName;
    if (item.hasOwnProperty('CategoryName')) return item.CategoryName;
    if (item.hasOwnProperty('CompanyName')) return item.CompanyName;
    if (item.hasOwnProperty('Country')) return item.Country;
    if (item.hasOwnProperty('Employee')) return item.Employee;
    
    // For date-based data
    if (item.hasOwnProperty('OrderDate') || item.hasOwnProperty('date')) {
      const date = item.OrderDate || item.date;
      return new Date(date).toLocaleDateString();
    }
    
    // Fallback to first numeric value
    const firstKey = Object.keys(item).find(key => 
      typeof item[key] === 'number' && key !== 'id'
    );
    return firstKey ? item[firstKey] : 'Value';
  });

  // Extract values from data
  const values = data.map(item => {
    // Find the first numeric value that's not an ID
    const valueKey = Object.keys(item).find(key => 
      typeof item[key] === 'number' && key !== 'id' && !key.toLowerCase().includes('id')
    );
    return valueKey ? item[valueKey] : 0;
  });

  // Chart data configuration
  const chartData = {
    labels: labels.slice(0, 10), // Limit to 10 items for better visualization
    datasets: [
      {
        label: 'Values',
        data: values.slice(0, 10),
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(199, 199, 199, 0.6)',
          'rgba(83, 102, 255, 0.6)',
          'rgba(40, 159, 64, 0.6)',
          'rgba(210, 99, 132, 0.6)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)',
          'rgba(199, 199, 199, 1)',
          'rgba(83, 102, 255, 1)',
          'rgba(40, 159, 64, 1)',
          'rgba(210, 99, 132, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Data Visualization',
      },
    },
  };

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
        return <Bar data={chartData} options={options} />;
      case 'line':
        return <Line data={chartData} options={options} />;
      case 'pie':
        return <Pie data={chartData} options={options} />;
      default:
        return <Bar data={chartData} options={options} />;
    }
  };

  return (
    <div className="chart-visualization">
      <div className="chart-controls">
        <h3>📊 Data Visualization</h3>
        <div className="chart-type-selector">
          <button 
            className={`chart-type-btn ${chartType === 'bar' ? 'active' : ''}`}
            onClick={() => onChartTypeChange('bar')}
          >
            Bar Chart
          </button>
          <button 
            className={`chart-type-btn ${chartType === 'line' ? 'active' : ''}`}
            onClick={() => onChartTypeChange('line')}
          >
            Line Graph
          </button>
          <button 
            className={`chart-type-btn ${chartType === 'pie' ? 'active' : ''}`}
            onClick={() => onChartTypeChange('pie')}
          >
            Pie Chart
          </button>
        </div>
      </div>
      <div className="chart-container">
        {renderChart()}
      </div>
    </div>
  );
};

export default ChartVisualization;