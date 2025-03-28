document.addEventListener("DOMContentLoaded", function () {
    // Initialize all charts
    let yearlyBarChart = null;
    let yearlyLineChart = null;
    let yearlyPieChart = null;
    let monthlyBarChart = null;
    let monthlyLineChart = null;
    let monthlyPieChart = null;

    // Function to initialize charts
    function initCharts() {
        // Yearly Consumption - Bar Chart
        const ctxYearlyBar = document.getElementById("yearlyBarChart").getContext("2d");
        yearlyBarChart = new Chart(ctxYearlyBar, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Yearly Consumption (MW)",
                    data: [],
                    backgroundColor: "rgba(76, 175, 80, 0.5)",
                    borderColor: "#4CAF50",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Energy Consumption (MW)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Year'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });

        // Yearly Consumption - Line Chart
        const ctxYearlyLine = document.getElementById("yearlyLineChart").getContext("2d");
        yearlyLineChart = new Chart(ctxYearlyLine, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Yearly Consumption (MW)",
                    data: [],
                    borderColor: "#2196F3",
                    backgroundColor: "rgba(33, 150, 243, 0.2)",
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Energy Consumption (MW)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Year'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });

        // Yearly Consumption - Pie Chart
        const ctxYearlyPie = document.getElementById("yearlyPieChart").getContext("2d");
        yearlyPieChart = new Chart(ctxYearlyPie, {
            type: "pie",
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        "#4CAF50",
                        "#2196F3",
                        "#FFC107",
                        "#9C27B0",
                        "#00BCD4"
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });

        // Monthly Consumption - Bar Chart
        const ctxMonthlyBar = document.getElementById("monthlyBarChart").getContext("2d");
        monthlyBarChart = new Chart(ctxMonthlyBar, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Monthly Consumption (MW)",
                    data: [],
                    backgroundColor: "rgba(255, 87, 51, 0.5)",
                    borderColor: "#FF5733",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Energy Consumption (MW)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });

        // Monthly Consumption - Line Chart
        const ctxMonthlyLine = document.getElementById("monthlyLineChart").getContext("2d");
        monthlyLineChart = new Chart(ctxMonthlyLine, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Monthly Consumption (MW)",
                    data: [],
                    borderColor: "#FF5733",
                    backgroundColor: "rgba(255, 87, 51, 0.2)",
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Energy Consumption (MW)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });

        // Monthly Consumption - Pie Chart
        const ctxMonthlyPie = document.getElementById("monthlyPieChart").getContext("2d");
        monthlyPieChart = new Chart(ctxMonthlyPie, {
            type: "pie",
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        "#FF5733",
                        "#4CAF50",
                        "#2196F3",
                        "#FFC107",
                        "#9C27B0"
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }

    // Initialize charts when page loads
    initCharts();

    // Function to fetch and update historical data
    async function fetchHistoricalData() {
        try {
            const response = await fetch('/get_historical_data');
            const data = await response.json();

            // Update Yearly Charts
            if (data.years && data.total_consumption) {
                // Bar Chart
                yearlyBarChart.data.labels = data.years;
                yearlyBarChart.data.datasets[0].data = data.total_consumption;
                yearlyBarChart.update();

                // Line Chart
                yearlyLineChart.data.labels = data.years;
                yearlyLineChart.data.datasets[0].data = data.total_consumption;
                yearlyLineChart.update();

                // Pie Chart
                yearlyPieChart.data.labels = data.years;
                yearlyPieChart.data.datasets[0].data = data.total_consumption;
                yearlyPieChart.update();
            }

            // Update Monthly Charts
            if (data.months && data.monthly_consumption) {
                // Bar Chart
                monthlyBarChart.data.labels = data.months;
                monthlyBarChart.data.datasets[0].data = data.monthly_consumption;
                monthlyBarChart.update();

                // Line Chart
                monthlyLineChart.data.labels = data.months;
                monthlyLineChart.data.datasets[0].data = data.monthly_consumption;
                monthlyLineChart.update();

                // Pie Chart
                monthlyPieChart.data.labels = data.months;
                monthlyPieChart.data.datasets[0].data = data.monthly_consumption;
                monthlyPieChart.update();
            }
        } catch (error) {
            console.error('Error fetching historical data:', error);
        }
    }

    // Fetch historical data when page loads
    fetchHistoricalData();
});
