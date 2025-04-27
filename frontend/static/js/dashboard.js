document.addEventListener("DOMContentLoaded", function () {
    // Prediction Charts
    let yearlyBarChart;
    let monthlyBarChart, monthlyLineChart, monthlyPieChart;

    // Previous Data Analysis Charts
    let yearlyUsageChart, monthlyUsageChart, urbanRuralChart, urbanSectorChart, ruralSectorChart, seasonUsageChart;

    const spinner = document.getElementById("spinner");

    function showSpinner() {
        spinner.style.display = "block";
    }

    function hideSpinner() {
        spinner.style.display = "none";
    }

    function initPredictionCharts() {
        const ctxMonthlyBar = document.getElementById("monthlyBarChart").getContext("2d");
        const ctxMonthlyLine = document.getElementById("monthlyLineChart").getContext("2d");
        const ctxMonthlyPie = document.getElementById("monthlyPieChart").getContext("2d");
        const ctxYearlyBar = document.getElementById("yearlyBarChart").getContext("2d");

        monthlyBarChart = new Chart(ctxMonthlyBar, {
            type: "bar",
            data: { labels: [], datasets: [{ label: "Monthly Consumption (MW)", data: [], backgroundColor: "rgba(255,87,51,0.5)", borderColor: "#FF5733", borderWidth: 1 }] },
            options: { responsive: true, animation: { duration: 1000 }, scales: { y: { beginAtZero: true } } }
        });

        monthlyLineChart = new Chart(ctxMonthlyLine, {
            type: "line",
            data: { labels: [], datasets: [{ label: "Monthly Consumption (MW)", data: [], borderColor: "#FF5733", backgroundColor: "rgba(255,87,51,0.2)", fill: true, tension: 0.3 }] },
            options: { responsive: true, animation: { duration: 1000 }, scales: { y: { beginAtZero: true } } }
        });

        monthlyPieChart = new Chart(ctxMonthlyPie, {
            type: "pie",
            data: { labels: [], datasets: [{ data: [], backgroundColor: ["#FF5733", "#4CAF50", "#2196F3", "#FFC107", "#9C27B0", "#00BCD4", "#8BC34A", "#E91E63", "#3F51B5", "#009688", "#FF9800", "#CDDC39"] }] },
            options: { responsive: true, animation: { duration: 1000 } }
        });

        yearlyBarChart = new Chart(ctxYearlyBar, {
            type: "bar",
            data: { labels: [], datasets: [{ label: "Total Yearly Consumption (MW)", data: [], backgroundColor: "rgba(76,175,80,0.5)", borderColor: "#4CAF50", borderWidth: 1 }] },
            options: { responsive: true, animation: { duration: 1000 }, scales: { y: { beginAtZero: true } } }
        });
    }

    async function loadPrediction(year) {
        try {
            showSpinner();
            const response = await fetch(`/predict_monthly_energy?year=${year}`);
            const data = await response.json();
            hideSpinner();

            const months = data.map(item => item.month);
            const predictedEnergy = data.map(item => item.predicted_energy);

            monthlyBarChart.data.labels = months;
            monthlyBarChart.data.datasets[0].data = predictedEnergy;
            monthlyBarChart.update();

            monthlyLineChart.data.labels = months;
            monthlyLineChart.data.datasets[0].data = predictedEnergy;
            monthlyLineChart.update();

            monthlyPieChart.data.labels = months;
            monthlyPieChart.data.datasets[0].data = predictedEnergy;
            monthlyPieChart.update();

            const totalEnergy = predictedEnergy.reduce((acc, val) => acc + val, 0);
            yearlyBarChart.data.labels = [year.toString()];
            yearlyBarChart.data.datasets[0].data = [totalEnergy];
            yearlyBarChart.update();

        } catch (error) {
            hideSpinner();
            console.error('Error fetching prediction data:', error);
        }
    }

    async function fetchPreviousData() {
        try {
            showSpinner();
            const response = await fetch('/previous_data_analysis');
            const data = await response.json();
            hideSpinner();

            if (data.error) {
                console.error('Error in previous data:', data.error);
                return;
            }

            // Yearly Usage Chart
            const years = Object.keys(data.yearly_usage);
            const yearlyUsage = Object.values(data.yearly_usage);
            yearlyUsageChart = new Chart(document.getElementById("yearlyUsageChart").getContext("2d"), {
                type: "bar",
                data: {
                    labels: years,
                    datasets: [{ label: "Yearly Usage (kWh)", data: yearlyUsage, backgroundColor: "#42a5f5" }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });

            // Monthly Usage Chart
            const months = Object.keys(data.monthly_usage);
            const monthlyUsage = Object.values(data.monthly_usage);
            monthlyUsageChart = new Chart(document.getElementById("monthlyUsageChart").getContext("2d"), {
                type: "bar",
                data: {
                    labels: months,
                    datasets: [{ label: "Monthly Usage (kWh)", data: monthlyUsage, backgroundColor: "#66bb6a" }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });

            // Urban vs Rural Usage Chart
            urbanRuralChart = new Chart(document.getElementById("urbanRuralChart").getContext("2d"), {
                type: "pie",
                data: {
                    labels: ["Urban", "Rural"],
                    datasets: [{
                        data: [data.sector_usage["Urban Usage (kWh)"], data.sector_usage["Rural Usage (kWh)"]],
                        backgroundColor: ["#ff7043", "#29b6f6"]
                    }]
                },
                options: { responsive: true }
            });

            // Urban Sector Usage Chart
            urbanSectorChart = new Chart(document.getElementById("urbanSectorChart").getContext("2d"), {
                type: "bar",
                data: {
                    labels: Object.keys(data.urban_sectors),
                    datasets: [{
                        label: "Urban Sector Usage (kWh)",
                        data: Object.values(data.urban_sectors),
                        backgroundColor: "#ab47bc"
                    }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });

            // Rural Sector Usage Chart
            ruralSectorChart = new Chart(document.getElementById("ruralSectorChart").getContext("2d"), {
                type: "bar",
                data: {
                    labels: Object.keys(data.rural_sectors),
                    datasets: [{
                        label: "Rural Sector Usage (kWh)",
                        data: Object.values(data.rural_sectors),
                        backgroundColor: "#26c6da"
                    }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });

            // Season Usage Chart
            seasonUsageChart = new Chart(document.getElementById("seasonUsageChart").getContext("2d"), {
                type: "pie",
                data: {
                    labels: Object.keys(data.season_usage),
                    datasets: [{
                        data: Object.values(data.season_usage),
                        backgroundColor: ["#ff8a65", "#81c784", "#64b5f6", "#ffd54f"]
                    }]
                },
                options: { responsive: true }
            });

        } catch (error) {
            hideSpinner();
            console.error('Error fetching previous data:', error);
        }
    }

    // Tab switch etc.
    const predictionTab = document.getElementById("predictionTab");
    const analysisTab = document.getElementById("analysisTab");
    const predictionSection = document.getElementById("predictionSection");
    const analysisSection = document.getElementById("analysisSection");

    predictionTab.addEventListener("click", function () {
        predictionTab.classList.add("active");
        analysisTab.classList.remove("active");
        predictionSection.style.display = "block";
        analysisSection.style.display = "none";
    });

    analysisTab.addEventListener("click", function () {
        analysisTab.classList.add("active");
        predictionTab.classList.remove("active");
        predictionSection.style.display = "none";
        analysisSection.style.display = "block";
    });

    const yearSelect = document.getElementById("yearSelect");
    yearSelect.addEventListener("change", function () {
        loadPrediction(this.value);
    });

    initPredictionCharts();
    loadPrediction(yearSelect.value);
    fetchPreviousData();
});