document.addEventListener("DOMContentLoaded", function () {
    // Initialize all charts
    let historicalYearlyChart = null;
    let historicalMonthlyChart = null;
    let sectorDistributionChart = null;
    let renewableEnergyChart = null;
    let peakDemandChart = null;
    let tempEnergyChart = null;

    // Function to initialize charts
    function initCharts() {
        // Historical Yearly Consumption
        const ctx1 = document.getElementById("historicalYearly").getContext("2d");
        historicalYearlyChart = new Chart(ctx1, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Yearly Consumption (MW)",
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

        // Historical Monthly Consumption
        const ctx2 = document.getElementById("historicalMonthly").getContext("2d");
        historicalMonthlyChart = new Chart(ctx2, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Monthly Consumption (MW)",
                    data: [],
                    borderColor: "#4CAF50",
                    backgroundColor: "rgba(76, 175, 80, 0.2)",
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

        // Sector Distribution
        const ctx3 = document.getElementById("sectorDistribution").getContext("2d");
        sectorDistributionChart = new Chart(ctx3, {
            type: "doughnut",
            data: {
                labels: ["Household", "Industrial", "Commercial", "Others"],
                datasets: [{
                    data: [],
                    backgroundColor: ["#FF5733", "#4CAF50", "#03A9F4", "#FFC107"],
                    hoverOffset: 4
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

        // Renewable vs Non-Renewable Energy
        const ctx4 = document.getElementById("renewableEnergy").getContext("2d");
        renewableEnergyChart = new Chart(ctx4, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Renewable Energy (MW)",
                    data: [],
                    backgroundColor: "rgba(76, 175, 80, 0.5)",
                    borderColor: "#4CAF50",
                    borderWidth: 1
                }, {
                    label: "Non-Renewable Energy (MW)",
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

        // Peak Demand Hours
        const ctx5 = document.getElementById("peakDemandHours").getContext("2d");
        peakDemandChart = new Chart(ctx5, {
            type: "line",
            data: {
                labels: ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                datasets: [{
                    label: "Average Demand (MW)",
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
                            text: 'Energy Demand (MW)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time of Day'
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

        // Temperature vs Energy Consumption
        const ctx6 = document.getElementById("tempEnergy").getContext("2d");
        tempEnergyChart = new Chart(ctx6, {
            type: "scatter",
            data: {
                datasets: [{
                    label: "Energy Consumption vs Temperature",
                    data: [],
                    backgroundColor: "rgba(76, 175, 80, 0.5)",
                    borderColor: "#4CAF50",
                    pointRadius: 4
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
                            text: 'Temperature (°C)'
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
    }

    // Initialize charts when page loads
    initCharts();

    // Function to fetch and update historical data
    async function fetchHistoricalData() {
        try {
            const response = await fetch('/get_historical_data');
            const data = await response.json();

            // Update all charts
            updateCharts(data);
        } catch (error) {
            console.error('Error fetching historical data:', error);
        }
    }

    // Function to update all charts with new data
    function updateCharts(data) {
        // Update Historical Yearly Consumption
        if (data.years) {
            historicalYearlyChart.data.labels = data.years;
            historicalYearlyChart.data.datasets[0].data = data.total_consumption;
            historicalYearlyChart.update();
        }

        // Update Historical Monthly Consumption
        if (data.months) {
            historicalMonthlyChart.data.labels = data.months;
            historicalMonthlyChart.data.datasets[0].data = data.monthly_consumption;
            historicalMonthlyChart.update();
        }

        // Update Sector Distribution
        if (data.sector_distribution) {
            sectorDistributionChart.data.datasets[0].data = Object.values(data.sector_distribution);
            sectorDistributionChart.data.labels = Object.keys(data.sector_distribution);
            sectorDistributionChart.update();
        }

        // Update Renewable vs Non-Renewable Energy
        if (data.renewable_data) {
            renewableEnergyChart.data.labels = data.renewable_data.years;
            renewableEnergyChart.data.datasets[0].data = data.renewable_data.renewable;
            renewableEnergyChart.data.datasets[1].data = data.renewable_data.non_renewable;
            renewableEnergyChart.update();
        }

        // Update Peak Demand Hours
        if (data.peak_demand) {
            peakDemandChart.data.datasets[0].data = data.peak_demand.values;
            peakDemandChart.update();
        }

        // Update Temperature vs Energy Consumption
        if (data.temp_energy) {
            tempEnergyChart.data.datasets[0].data = data.temp_energy.map(point => ({
                x: point.temp,
                y: point.energy
            }));
            tempEnergyChart.update();
        }
    }

    // Fetch historical data when page loads
    fetchHistoricalData();
});
