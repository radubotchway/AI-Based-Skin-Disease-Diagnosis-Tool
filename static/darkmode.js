// Function to toggle dark mode
function toggleDarkMode() {
    // Toggle dark mode class on body
    document.body.classList.toggle('dark-mode');

    // Store dark mode preference in localStorage
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
}

// Function to initialize dark mode based on user preference
function initializeDarkMode() {
    // Check if dark mode preference is stored in localStorage
    const darkMode = localStorage.getItem('darkMode');

    // If dark mode preference exists and is true, enable dark mode
    if (darkMode && darkMode === 'true') {
        document.body.classList.add('dark-mode');
    }
}

// Initialize dark mode
initializeDarkMode();
