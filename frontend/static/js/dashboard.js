document.addEventListener("DOMContentLoaded", function () {
    const ctxConsumption = document.getElementById("monthlyConsumptionChart").getContext("2d");
    const ctxSector = document.getElementById("sectorDistributionChart").getContext("2d");

    // 🔹 Placeholder energy data (replace with API call)
    const energyData = {
        total: [12000, 12500, 13000, 13500, 14000, 14500, 15000, 15200, 15500, 15800, 16000, 16200],
        urban: [7000, 7200, 7400, 7600, 7800, 8000, 8200, 8300, 8500, 8700, 8900, 9100],
        rural: [5000, 5300, 5600, 5900, 6200, 6500, 6800, 6900, 7000, 7100, 7100, 7100]
    };

    // 🔹 Update stats on page load
    function updateStats(monthIndex) {
        document.getElementById("totalEnergy").textContent = `${energyData.total[monthIndex]} MW`;
        document.getElementById("urbanEnergy").textContent = `${energyData.urban[monthIndex]} MW`;
        document.getElementById("ruralEnergy").textContent = `${energyData.rural[monthIndex]} MW`;
    }

    // 🔹 Monthly consumption line chart
    const monthlyConsumptionChart = new Chart(ctxConsumption, {
        type: "line",
        data: {
            labels: [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            datasets: [{
                label: "Total Energy Consumption (MW)",
                data: energyData.total,
                borderColor: "#FF5733",
                backgroundColor: "rgba(255, 87, 51, 0.2)",
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: "top" },
                title: { display: true, text: "Monthly Energy Consumption" }
            }
        }
    });

    // 🔹 Sector-wise distribution pie chart
    const sectorDistributionChart = new Chart(ctxSector, {
        type: "doughnut",
        data: {
            labels: ["Household", "Industrial", "Commercial", "Others"],
            datasets: [{
                label: "Sector-wise Energy Distribution",
                data: [3000, 4000, 2000, 1000],
                backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: "top" },
                title: { display: true, text: "Sector-wise Energy Distribution" }
            }
        }
    });

    // 🔹 Handle month selection change
    document.getElementById("monthDropdown").addEventListener("change", function (event) {
        const monthIndex = event.target.selectedIndex;
        updateStats(monthIndex);
    });

    // 🔹 Set initial values
    updateStats(0);
});
