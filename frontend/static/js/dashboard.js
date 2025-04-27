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

            // 🚫 No yearly line chart
            // 🚫 No yearly pie chart

        } catch (error) {
            hideSpinner();
            console.error('Error fetching prediction data:', error);
        }
    }

    async function fetchPreviousData() {
        // (Keep your fetchPreviousData same — no changes needed for old analysis charts)
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
