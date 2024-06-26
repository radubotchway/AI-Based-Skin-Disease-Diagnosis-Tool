// upload-script.js

// Get drop area
const dropArea = document.getElementById('drop-area');
const dropLabel = document.getElementById('drop-label');
const imageInput = document.getElementById('image');
const previewContainer = document.getElementById('preview-container');

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
imageInput.addEventListener('change', handleFiles, false);

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
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();

            reader.onload = function() {
                const img = document.createElement('img');
                img.src = reader.result;
                previewContainer.innerHTML = ''; // Clear previous preview
                previewContainer.appendChild(img);
            }

            reader.readAsDataURL(file);
        }
    }
}
