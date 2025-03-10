document.addEventListener("DOMContentLoaded", function() {
    // Sample data for Total Energy Consumption Trend
    const consumptionTrendData = {
        labels: ['January', 'February', 'March', 'April', 'May', 'June'],
        datasets: [{
            label: 'Total Energy Consumption (MW)',
            data: [12000, 12500, 13000, 13500, 14000, 14500],
            borderColor: '#FF5733',
            backgroundColor: 'rgba(255, 87, 51, 0.2)',
            fill: true,
        }]
    };

    const consumptionTrendConfig = {
        type: 'line',
        data: consumptionTrendData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Total Energy Consumption Trend'
                }
            }
        },
    };

    new Chart(
        document.getElementById('consumptionTrend'),
        consumptionTrendConfig
    );

    // Sample data for Sector-wise Energy Distribution
    const sectorDistributionData = {
        labels: ['Household', 'Industrial', 'Commercial', 'Others'],
        datasets: [{
            label: 'Energy Distribution (MW)',
            data: [3000, 4000, 2000, 1000],
            backgroundColor: [
                '#FF6384',
                '#36A2EB',
                '#FFCE56',
                '#4BC0C0'
            ],
            hoverOffset: 4
        }]
    };

    const sectorDistributionConfig = {
        type: 'doughnut',
        data: sectorDistributionData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Sector-wise Energy Distribution'
                }
            }
        },
    };

    new Chart(
        document.getElementById('sectorDistribution'),
        sectorDistributionConfig
    );
});
