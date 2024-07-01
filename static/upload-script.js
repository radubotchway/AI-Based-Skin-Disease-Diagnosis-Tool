// Get elements
const dropArea = document.getElementById('drop-area');
const dropLabel = document.getElementById('drop-label');
const imageInput = document.getElementById('image');
const previewContainer = document.getElementById('preview-container');
const submitButton = document.getElementById('submit-button');
const predictionResult = document.getElementById('prediction-result');

let selectedFile;

// Prevent default behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when file is dragged over
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

// Clicking on drop area triggers file input click
dropLabel.addEventListener('click', () => {
    imageInput.click();
});

// Handle file input change
imageInput.addEventListener('change', (event) => {
    handleFiles(event.target.files);
}, false);

function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
}

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

function handleDrop(event) {
    const dt = event.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        if (selectedFile.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function() {
                const img = document.createElement('img');
                img.src = reader.result;
                img.style.maxWidth = '100%'; // Adjust the image size to fit the container
                previewContainer.innerHTML = ''; // Clear previous preview
                previewContainer.appendChild(img);
            }
            reader.readAsDataURL(selectedFile);
        }
    }
}
