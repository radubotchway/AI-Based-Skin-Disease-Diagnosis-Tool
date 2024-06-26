document.getElementById('submit-symptoms').addEventListener('click', function() {
    // Get user inputs
    const symptomDescription = document.getElementById('symptom-description').value;
    const duration = document.getElementById('duration').value;
    const itchingLevel = document.getElementById('itching').value;
    const painLevel = document.getElementById('pain').value;

    // You can do further processing or send this data to the server for analysis
    console.log('Symptom Description:', symptomDescription);
    console.log('Duration:', duration);
    console.log('Itching Level:', itchingLevel);
    console.log('Pain Level:', painLevel);

    // Optionally, you can redirect the user to another page after submission
    // window.location.href = 'confirmation.html';
});
