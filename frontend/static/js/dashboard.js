document.addEventListener("DOMContentLoaded", function () {
    // Prediction Charts
    let yearlyBarChart, yearlyLineChart, yearlyPieChart;
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
        const ctxYearlyLine = document.getElementById("yearlyLineChart").getContext("2d");
        const ctxYearlyPie = document.getElementById("yearlyPieChart").getContext("2d");

        monthlyBarChart = new Chart(ctxMonthlyBar, { type: "bar", data: { labels: [], datasets: [{ label: "Monthly Consumption (MW)", data: [], backgroundColor: "rgba(255,87,51,0.5)", borderColor: "#FF5733", borderWidth: 1 }] }, options: { responsive: true, scales: { y: { beginAtZero: true } } } });
        monthlyLineChart = new Chart(ctxMonthlyLine, { type: "line", data: { labels: [], datasets: [{ label: "Monthly Consumption (MW)", data: [], borderColor: "#FF5733", backgroundColor: "rgba(255,87,51,0.2)", fill: true, tension: 0.3 }] }, options: { responsive: true, scales: { y: { beginAtZero: true } } } });
        monthlyPieChart = new Chart(ctxMonthlyPie, { type: "pie", data: { labels: [], datasets: [{ data: [], backgroundColor: ["#FF5733", "#4CAF50", "#2196F3", "#FFC107", "#9C27B0", "#00BCD4", "#8BC34A", "#E91E63", "#3F51B5", "#009688", "#FF9800", "#CDDC39"] }] }, options: { responsive: true } });

        yearlyBarChart = new Chart(ctxYearlyBar, { type: "bar", data: { labels: [], datasets: [{ label: "Total Yearly Consumption (MW)", data: [], backgroundColor: "rgba(76,175,80,0.5)", borderColor: "#4CAF50", borderWidth: 1 }] }, options: { responsive: true, scales: { y: { beginAtZero: true } } } });
        yearlyLineChart = new Chart(ctxYearlyLine, { type: "line", data: { labels: [], datasets: [{ label: "Total Yearly Consumption (MW)", data: [], borderColor: "#2196F3", backgroundColor: "rgba(33,150,243,0.2)", fill: true, tension: 0.3 }] }, options: { responsive: true, scales: { y: { beginAtZero: true } } } });
        yearlyPieChart = new Chart(ctxYearlyPie, { type: "pie", data: { labels: [], datasets: [{ data: [], backgroundColor: ["#4CAF50", "#2196F3", "#FFC107", "#9C27B0"] }] }, options: { responsive: true } });
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

            yearlyLineChart.data.labels = [year.toString()];
            yearlyLineChart.data.datasets[0].data = [totalEnergy];
            yearlyLineChart.update();

            yearlyPieChart.data.labels = [year.toString()];
            yearlyPieChart.data.datasets[0].data = [totalEnergy];
            yearlyPieChart.update();

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

            if (yearlyUsageChart) yearlyUsageChart.destroy();
            if (monthlyUsageChart) monthlyUsageChart.destroy();
            if (urbanRuralChart) urbanRuralChart.destroy();
            if (urbanSectorChart) urbanSectorChart.destroy();
            if (ruralSectorChart) ruralSectorChart.destroy();
            if (seasonUsageChart) seasonUsageChart.destroy();

            yearlyUsageChart = new Chart(document.getElementById('yearlyUsageChart'), {
                type: 'bar',
                data: { labels: Object.keys(data.yearly_usage), datasets: [{ label: "Total Usage (kWh)", data: Object.values(data.yearly_usage), backgroundColor: '#4CAF50' }] },
                options: { responsive: true }
            });

            monthlyUsageChart = new Chart(document.getElementById('monthlyUsageChart'), {
                type: 'line',
                data: { labels: Object.keys(data.monthly_usage), datasets: [{ label: "Monthly Usage (kWh)", data: Object.values(data.monthly_usage), borderColor: '#2196F3', fill: true }] },
                options: { responsive: true }
            });

            urbanRuralChart = new Chart(document.getElementById('urbanRuralChart'), {
                type: 'pie',
                data: { labels: ['Urban Usage', 'Rural Usage'], datasets: [{ data: Object.values(data.sector_usage), backgroundColor: ['#FF5733', '#FFC107'] }] },
                options: { responsive: true }
            });

            urbanSectorChart = new Chart(document.getElementById('urbanSectorChart'), {
                type: 'doughnut',
                data: { labels: Object.keys(data.urban_sectors), datasets: [{ data: Object.values(data.urban_sectors), backgroundColor: ['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B'] }] },
                options: { responsive: true }
            });

            ruralSectorChart = new Chart(document.getElementById('ruralSectorChart'), {
                type: 'doughnut',
                data: { labels: Object.keys(data.rural_sectors), datasets: [{ data: Object.values(data.rural_sectors), backgroundColor: ['#FF9800', '#FF5722', '#795548', '#9E9E9E'] }] },
                options: { responsive: true }
            });

            seasonUsageChart = new Chart(document.getElementById('seasonUsageChart'), {
                type: 'bar',
                data: { labels: Object.keys(data.season_usage), datasets: [{ label: "Seasonal Usage (kWh)", data: Object.values(data.season_usage), backgroundColor: '#3F51B5' }] },
                options: { responsive: true }
            });

        } catch (error) {
            hideSpinner();
            console.error('Error fetching previous data:', error);
        }
    }

    // Handle Tabs
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

    // Year change for prediction
    const yearSelect = document.getElementById("yearSelect");
    yearSelect.addEventListener("change", function () {
        loadPrediction(this.value);
    });

    // Initialize
    initPredictionCharts();
    loadPrediction(yearSelect.value);
    fetchPreviousData();
});
