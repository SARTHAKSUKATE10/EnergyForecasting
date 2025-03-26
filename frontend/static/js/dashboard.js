document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("fetchData").addEventListener("click", function () {
        const month = document.getElementById("month").value;
        const year = document.getElementById("year").value;
        const season = document.getElementById("season").value;
        const period = document.getElementById("period").value;
        const temp = document.getElementById("temp").value;
        const humidity = document.getElementById("humidity").value;
        const rainfall = document.getElementById("rainfall").value;
        const population = document.getElementById("population").value;

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                features: [parseInt(year + month)],
                season: season,
                period: period,
                temp: parseFloat(temp),
                rainfall: parseFloat(rainfall),
                humidity: parseFloat(humidity),
                population: parseInt(population)
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("totalEnergy").textContent = data.total_energy.toFixed(2) + " MW";
            document.getElementById("peakDemand").textContent = data.peak_demand.toFixed(2) + " MW";
            document.getElementById("renewableEnergy").textContent = data.renewable_percentage.toFixed(2) + "%";

            // Update Charts
            updateCharts(data);
        })
        .catch(error => console.error("Error fetching data:", error));
    });

    function updateCharts(data) {
        const ctx1 = document.getElementById("consumptionTrend").getContext("2d");
        new Chart(ctx1, {
            type: "line",
            data: {
                labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                datasets: [{
                    label: "Total Energy Consumption (MW)",
                    data: data.monthly_trend,
                    borderColor: "#FF5733",
                    backgroundColor: "rgba(255, 87, 51, 0.2)",
                    fill: true
                }]
            }
        });

        const ctx2 = document.getElementById("sectorDistribution").getContext("2d");
        new Chart(ctx2, {
            type: "doughnut",
            data: {
                labels: ["Household", "Industrial", "Commercial", "Others"],
                datasets: [{
                    data: data.sector_distribution,
                    backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"]
                }]
            }
        });
    }
});



