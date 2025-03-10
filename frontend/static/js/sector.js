document.addEventListener("DOMContentLoaded", function() {
    // Function to fetch and update sector data
    async function updateSectorData() {
        try {
            const response = await fetch('/api/sector-data'); // Replace with your actual API endpoint
            if (!response.ok) throw new Error('Failed to fetch sector data');
            const data = await response.json();

            // Update the DOM elements with fetched data
            document.querySelector('.industrial .consumption').innerText = `Consumption: ${data.industrial} MW`;
            document.querySelector('.industrial .chart-bar').style.width = `${data.industrialPercentage}%`;

            document.querySelector('.residential .consumption').innerText = `Consumption: ${data.residential} MW`;
            document.querySelector('.residential .chart-bar').style.width = `${data.residentialPercentage}%`;

            document.querySelector('.commercial .consumption').innerText = `Consumption: ${data.commercial} MW`;
            document.querySelector('.commercial .chart-bar').style.width = `${data.commercialPercentage}%`;

            document.querySelector('.agricultural .consumption').innerText = `Consumption: ${data.agricultural} MW`;
            document.querySelector('.agricultural .chart-bar').style.width = `${data.agriculturalPercentage}%`;

            document.querySelector('.total-value').innerText = `${data.total} MW`;
        } catch (error) {
            console.error('Error updating sector data:', error);
        }
    }

    // Initial data fetch
    updateSectorData();

    // Call updateSectorData every 5 seconds to fetch real-time data
    setInterval(updateSectorData, 5000);

    // Set interval to update data every 5 minutes
    setInterval(updateSectorData, 300000);
});
