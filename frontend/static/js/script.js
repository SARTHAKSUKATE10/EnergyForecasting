document.addEventListener("DOMContentLoaded", function() {
    console.log("DOM fully loaded, script.js is running.");

    document.getElementById("predictionForm").addEventListener("submit", async function(event) {
        event.preventDefault(); // Prevent default form submission (page reload)

        try {
            // Get and validate the date input
            const dateElem = document.getElementById("date");
            if (!dateElem.value) {
                throw new Error("Please select a date.");
            }
            const dateValue = dateElem.value;
            // Convert date to a timestamp (in seconds)
            const timestamp = new Date(dateValue).getTime() / 1000;
            console.log("Date:", dateValue, "Timestamp:", timestamp);

            // Get the Festival value from dropdown
            const festivalVal = parseInt(document.getElementById("Festival").value);

            // Build the payload using form values
            const data = {
                features: [timestamp, festivalVal],
                season: document.getElementById("season").value,
                period: document.getElementById("period").value,
                temp: parseFloat(document.getElementById("temp").value),
                rainfall: parseFloat(document.getElementById("rainfall").value) || 0,
                humidity: parseFloat(document.getElementById("humidity").value),
                population: parseInt(document.getElementById("population").value),
                festival: festivalVal
            };

            console.log("Payload:", data);

            // Send POST request to the /predict endpoint
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error("Response error:", errorData);
                throw new Error(errorData.error || "Server error");
            }

            const result = await response.json();
            console.log("API result:", result);
            displayPrediction(result);
        } catch (error) {
            console.error("Error fetching prediction:", error);
            document.getElementById("predictionOutput").innerText = "⚠️ Error: " + error.message;
        }
    });

    function displayPrediction(result) {
        // Check that the result has the expected keys
        if (!result["Total Energy"]) {
            document.getElementById("predictionOutput").innerText = "⚠️ Unexpected response format.";
            return;
        }

        // Clear the previous output
        document.getElementById("predictionOutput").innerText = "";

        // Build card-based HTML output
        const outputHTML = `
            <div class="card">
                <h5>⚡ Total Energy</h5>
                <p>${result["Total Energy"].toFixed(2)} kWh</p>
            </div>
            <div class="card">
                <h5>🏙️ Urban Usage</h5>
                <p>${result["Urban Usage"].toFixed(2)} kWh</p>
                <h6>Distribution:</h6>
                <ul>
                    <li>🏠 Household: ${result["Urban Distribution"]["Urban Household"].toFixed(2)} kWh</li>
                    <li>🏭 Industrial: ${result["Urban Distribution"]["Urban Industrial"].toFixed(2)} kWh</li>
                    <li>🏢 Commercial: ${result["Urban Distribution"]["Urban Commercial"].toFixed(2)} kWh</li>
                    <li>🌐 Others: ${result["Urban Distribution"]["Urban Others"].toFixed(2)} kWh</li>
                </ul>
            </div>
            <div class="card">
                <h5>🌾 Rural Usage</h5>
                <p>${result["Rural Usage"].toFixed(2)} kWh</p>
                <h6>Distribution:</h6>
                <ul>
                    <li>🏠 Household: ${result["Rural Distribution"]["Rural Household"].toFixed(2)} kWh</li>
                    <li>🏭 Industrial: ${result["Rural Distribution"]["Rural Industrial"].toFixed(2)} kWh</li>
                    <li>🏢 Commercial: ${result["Rural Distribution"]["Rural Commercial"].toFixed(2)} kWh</li>
                    <li>🌐 Others: ${result["Rural Distribution"]["Rural Others"].toFixed(2)} kWh</li>
                </ul>
            </div>
        `;

        // Make the energy-consumption-result section visible
        const energyResultSection = document.getElementById("energyConsumptionResult");
        energyResultSection.classList.add("active");
        energyResultSection.style.display = "block";

        // Update the content of the energy-consumption-result section
        energyResultSection.innerHTML = outputHTML;

        // Sample data for Total Energy Consumption Trend
        const totalEnergyTrendData = {
            labels: ['January', 'February', 'March', 'April', 'May', 'June'],
            datasets: [{
                label: 'Total Energy Consumption (kWh)',
                data: [12000, 15000, 13000, 16000, 17000, 18000],
                borderColor: '#FF5733',
                backgroundColor: 'rgba(255, 87, 51, 0.2)',
                fill: true,
            }]
        };

        const totalEnergyTrendConfig = {
            type: 'line',
            data: totalEnergyTrendData,
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
            document.getElementById('totalEnergyTrendChart'),
            totalEnergyTrendConfig
        );

        // Sample data for Sector-wise Energy Distribution
        const sectorWiseDistributionData = {
            labels: ['Household', 'Industrial', 'Commercial', 'Others'],
            datasets: [{
                label: 'Energy Distribution (kWh)',
                data: [5000, 7000, 3000, 2000],
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0'
                ],
                hoverOffset: 4
            }]
        };

        const sectorWiseDistributionConfig = {
            type: 'doughnut',
            data: sectorWiseDistributionData,
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
            document.getElementById('sectorWiseDistributionChart'),
            sectorWiseDistributionConfig
        );
    }
});
