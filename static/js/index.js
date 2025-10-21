function generateTable(leaderboardData) {
    const tbody = document.querySelector('#leaderboard tbody');
    tbody.innerHTML = ""; // Clear existing rows

    // Separate human_expert and random_guess entries from others
    const humanExperts = leaderboardData.leaderboardData.filter(entry => entry.info.type === 'human_expert');
    const randomGuesses = leaderboardData.leaderboardData.filter(entry => entry.info.type === 'random_guess');
    const others = leaderboardData.leaderboardData.filter(entry => 
        entry.info.type !== 'human_expert' && entry.info.type !== 'random_guess'
    );

    // Sort only the non-fixed entries based on Avg["Test"] in descending order
    others.sort((a, b) => {
        let avgA = parseFloat(a.Avg?.["Test"]) || 0;
        let avgB = parseFloat(b.Avg?.["Test"]) || 0;
        return avgB - avgA; // Sort descending
    });

    // Function to create and append table rows
    function appendRow(entry, isDivider = false) {
        const totalColumns = 10; // Adjust this based on the actual number of columns

        if (isDivider) {
            const dividerRow = document.createElement('tr');
            dividerRow.innerHTML = `<td colspan="${totalColumns}" style="border-bottom: 4px solid black;"></td>`;
            tbody.appendChild(dividerRow);
            return;
        }

        const row = document.createElement('tr');

        if (entry.info.type === 'human_expert') {
            row.classList.add('human_expert');
        } else if (entry.info.type === 'random_guess') {
            row.classList.add('random_guess');
        } else if (entry.info.type === 'open_source') {
            row.classList.add('open_source');
        } else if (entry.info.type === 'proprietary') {
            row.classList.add('proprietary');
        } else if (entry.info.type === 'open_access') {
            row.classList.add('open_access');
        }  else if (entry.info.type === 'fine_tuned') {
            row.classList.add('fine_tuned');
        }

        // Assign medals to the top 3 performers in the sorted list (excluding fixed entries)
        let medal = "";
        if (entry.rank === 0) {
            medal = " 🥇";  // Gold Medal
        } else if (entry.rank === 1) {
            medal = " 🥈";  // Silver Medal
        } else if (entry.rank === 2) {
            medal = " 🥉";  // Bronze Medal
        }

        // Name with optional link
        let nameCell = `<td>${entry.info.link ? `<a href="${entry.info.link}">${entry.info.name}${medal}</a>` : entry.info.name + medal}</td>`;

        // Size
        let sizeCell = `<td>${entry.info.size || '-'}</td>`;

        // Sound, Music, Speech, Avg (Test-mini and Test)
        let soundMini = `<td>${entry.Sound?.["Test-mini"] || '-'}</td>`;
        let soundTest = `<td>${entry.Sound?.["Test"] || '-'}</td>`;
        let musicMini = `<td>${entry.Music?.["Test-mini"] || '-'}</td>`;
        let musicTest = `<td>${entry.Music?.["Test"] || '-'}</td>`;
        let speechMini = `<td>${entry.Speech?.["Test-mini"] || '-'}</td>`;
        let speechTest = `<td>${entry.Speech?.["Test"] || '-'}</td>`;
        let avgMini = `<td>${entry.Avg?.["Test-mini"] || '-'}</td>`;
        let avgTest = `<td>${entry.Avg?.["Test"] || '-'}</td>`;

        // Append all cells into the row
        row.innerHTML = `${nameCell}${sizeCell}${soundMini}${soundTest}${musicMini}${musicTest}${speechMini}${speechTest}${avgMini}${avgTest}`;

        tbody.appendChild(row);
    }

    // Append random guess entries first
    randomGuesses.forEach(entry => appendRow(entry));

    // Append human experts immediately after random guess (no separator)
    humanExperts.forEach(entry => appendRow(entry));

    // Insert a single bold line before sorted entries
    if (others.length > 0) {
        appendRow(null, true);
    }

    // Append sorted non-fixed entries
    others.forEach((entry, index) => {
        entry.rank = index; // Track index for medal assignment
        appendRow(entry);
    });
}

function generateTableV15(leaderboardData) {
    const tbody = document.querySelector('#leaderboard-v15 tbody');
    tbody.innerHTML = "";

    const humanExperts = leaderboardData.leaderboardData.filter(entry => entry.info.type === 'human_expert');
    const randomGuesses = leaderboardData.leaderboardData.filter(entry => entry.info.type === 'random_guess');
    const others = leaderboardData.leaderboardData.filter(entry => 
        entry.info.type !== 'human_expert' && entry.info.type !== 'random_guess'
    );

    others.sort((a, b) => {
        let avgA = parseFloat(a.Avg?.["Test"]) || 0;
        let avgB = parseFloat(b.Avg?.["Test"]) || 0;
        return avgB - avgA;
    });

    function appendRow(entry, isDivider = false) {
        const totalColumns = 10;

        if (isDivider) {
            const dividerRow = document.createElement('tr');
            dividerRow.innerHTML = `<td colspan="${totalColumns}" style="border-bottom: 4px solid black;"></td>`;
            tbody.appendChild(dividerRow);
            return;
        }

        const row = document.createElement('tr');
        if (entry.info.type === 'human_expert') row.classList.add('human_expert');
        else if (entry.info.type === 'random_guess') row.classList.add('random_guess');
        else if (entry.info.type === 'open_source') row.classList.add('open_source');
        else if (entry.info.type === 'proprietary') row.classList.add('proprietary');
        else if (entry.info.type === 'open_access') row.classList.add('open_access');
        else if (entry.info.type === 'fine_tuned') row.classList.add('fine_tuned');

        let medal = "";
        if (entry.rank === 0) medal = " 🥇";
        else if (entry.rank === 1) medal = " 🥈";
        else if (entry.rank === 2) medal = " 🥉";

        let nameCell = `<td>${entry.info.link ? `<a href="${entry.info.link}">${entry.info.name}${medal}</a>` : entry.info.name + medal}</td>`;
        let sizeCell = `<td>${entry.info.size || '-'}</td>`;
        let soundMini = `<td>${entry.Sound?.["Test-mini"] || '-'}</td>`;
        let soundTest = `<td>${entry.Sound?.["Test"] || '-'}</td>`;
        let musicMini = `<td>${entry.Music?.["Test-mini"] || '-'}</td>`;
        let musicTest = `<td>${entry.Music?.["Test"] || '-'}</td>`;
        let speechMini = `<td>${entry.Speech?.["Test-mini"] || '-'}</td>`;
        let speechTest = `<td>${entry.Speech?.["Test"] || '-'}</td>`;
        let avgMini = `<td>${entry.Avg?.["Test-mini"] || '-'}</td>`;
        let avgTest = `<td>${entry.Avg?.["Test"] || '-'}</td>`;

        row.innerHTML = `${nameCell}${sizeCell}${soundMini}${soundTest}${musicMini}${musicTest}${speechMini}${speechTest}${avgMini}${avgTest}`;
        tbody.appendChild(row);
    }

    randomGuesses.forEach(entry => appendRow(entry));
    humanExperts.forEach(entry => appendRow(entry));
    if (others.length > 0) appendRow(null, true);
    others.forEach((entry, index) => {
        entry.rank = index;
        appendRow(entry);
    });
}

function generateTableV15Parsed(leaderboardData) {
    const tbody = document.querySelector('#leaderboard-v15-parsed tbody');
    tbody.innerHTML = "";

    const humanExperts = leaderboardData.leaderboardData.filter(entry => entry.info.type === 'human_expert');
    const randomGuesses = leaderboardData.leaderboardData.filter(entry => entry.info.type === 'random_guess');
    const others = leaderboardData.leaderboardData.filter(entry => 
        entry.info.type !== 'human_expert' && entry.info.type !== 'random_guess'
    );

    others.sort((a, b) => {
        let avgA = parseFloat(a.Avg?.["Test"]) || 0;
        let avgB = parseFloat(b.Avg?.["Test"]) || 0;
        return avgB - avgA;
    });

    function appendRow(entry, isDivider = false) {
        const totalColumns = 10;

        if (isDivider) {
            const dividerRow = document.createElement('tr');
            dividerRow.innerHTML = `<td colspan="${totalColumns}" style="border-bottom: 4px solid black;"></td>`;
            tbody.appendChild(dividerRow);
            return;
        }

        const row = document.createElement('tr');
        if (entry.info.type === 'human_expert') row.classList.add('human_expert');
        else if (entry.info.type === 'random_guess') row.classList.add('random_guess');
        else if (entry.info.type === 'open_source') row.classList.add('open_source');
        else if (entry.info.type === 'proprietary') row.classList.add('proprietary');
        else if (entry.info.type === 'open_access') row.classList.add('open_access');
        else if (entry.info.type === 'fine_tuned') row.classList.add('fine_tuned');

        let medal = "";
        if (entry.rank === 0) medal = " 🥇";
        else if (entry.rank === 1) medal = " 🥈";
        else if (entry.rank === 2) medal = " 🥉";

        let nameCell = `<td>${entry.info.link ? `<a href="${entry.info.link}">${entry.info.name}${medal}</a>` : entry.info.name + medal}</td>`;
        let sizeCell = `<td>${entry.info.size || '-'}</td>`;
        let soundMini = `<td>${entry.Sound?.["Test-mini"] || '-'}</td>`;
        let soundTest = `<td>${entry.Sound?.["Test"] || '-'}</td>`;
        let musicMini = `<td>${entry.Music?.["Test-mini"] || '-'}</td>`;
        let musicTest = `<td>${entry.Music?.["Test"] || '-'}</td>`;
        let speechMini = `<td>${entry.Speech?.["Test-mini"] || '-'}</td>`;
        let speechTest = `<td>${entry.Speech?.["Test"] || '-'}</td>`;
        let avgMini = `<td>${entry.Avg?.["Test-mini"] || '-'}</td>`;
        let avgTest = `<td>${entry.Avg?.["Test"] || '-'}</td>`;

        row.innerHTML = `${nameCell}${sizeCell}${soundMini}${soundTest}${musicMini}${musicTest}${speechMini}${speechTest}${avgMini}${avgTest}`;
        tbody.appendChild(row);
    }

    randomGuesses.forEach(entry => appendRow(entry));
    humanExperts.forEach(entry => appendRow(entry));
    if (others.length > 0) appendRow(null, true);
    others.forEach((entry, index) => {
        entry.rank = index;
        appendRow(entry);
    });
}

function loadV15ParsedLeaderboard() {
    fetch('./leaderboard_data_v15_parsed.json')  // Path to new leaderboard JSON
        .then(response => response.json())
        .then(data => generateTableV15Parsed(data))
        .catch(error => console.error('Error loading v1.5 JSON data:', error));
}

function loadV15Leaderboard() {
    fetch('./leaderboard_data_v15.json')  // Path to new leaderboard JSON
        .then(response => response.json())
        .then(data => generateTableV15(data))
        .catch(error => console.error('Error loading v1.5 JSON data:', error));
}


// Function to load JSON and then generate the table
function loadJSONAndGenerateTable() {
    fetch('./leaderboard_data.json')  // Replace this with the path to your JSON file
        .then(response => response.json())
        .then(data => generateTable(data))
        .catch(error => console.error('Error loading the JSON data:', error));
}

// Call the function on page load
// document.addEventListener('DOMContentLoaded', loadJSONAndGenerateTable);
document.addEventListener('DOMContentLoaded', () => {
    loadJSONAndGenerateTable();  // Old table
    loadV15Leaderboard();        // New MMAUV1.5 leaderboard
    loadV15ParsedLeaderboard(); // New MMAUV1.5 leaderboard parsed
});
