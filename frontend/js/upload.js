/**
 * Diabetic Retinopathy Screening System
 * Phase 2 - Retinal Image Upload Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const dropZonePrompt = document.getElementById('dropZonePrompt');
  const previewWrapper = document.getElementById('previewWrapper');
  const imagePreview = document.getElementById('imagePreview');
  const fileNameDisplay = document.getElementById('fileNameDisplay');
  const fileSizeDisplay = document.getElementById('fileSizeDisplay');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');

  // Alert Elements
  const validationAlert = document.getElementById('validationAlert');
  const alertTitle = document.getElementById('alertTitle');
  const alertMessage = document.getElementById('alertMessage');
  const dismissAlertBtn = document.getElementById('dismissAlertBtn');

  // Allowed file extensions & MIME types
  const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];
  const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];

  let selectedFile = null;

  // Trigger file browser when clicking "Choose Image" button or drop zone prompt
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZonePrompt.addEventListener('click', () => {
    fileInput.click();
  });

  // Drag and drop event handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  // Handle file input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  // File selection processor and validator
  function handleFileSelection(file) {
    hideAlert();

    if (!file) {
      showError('No File Selected', 'Please select a valid fundus image file.');
      resetUploadState();
      return;
    }

    const fileName = file.name.toLowerCase();
    const isExtensionValid = ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
    const isTypeValid = ALLOWED_MIME_TYPES.includes(file.type);

    if (!isExtensionValid && !isTypeValid) {
      showError(
        'Unsupported File Format',
        `"${file.name}" is not supported. Please select a JPG, JPEG, or PNG image.`
      );
      resetUploadState();
      return;
    }

    // Read and display preview
    const reader = new FileReader();
    reader.onload = (e) => {
      selectedFile = file;
      imagePreview.src = e.target.result;
      fileNameDisplay.textContent = file.name;
      fileSizeDisplay.textContent = formatFileSize(file.size);

      // Toggle visibility
      dropZonePrompt.classList.add('d-none');
      previewWrapper.classList.remove('d-none');

      // Enable Analyze button
      analyzeBtn.disabled = false;
    };

    reader.onerror = () => {
      showError('File Error', 'Unable to read the selected file. Please try another image.');
      resetUploadState();
    };

    reader.readAsDataURL(file);
  }

  // Remove file and reset state
  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUploadState();
    hideAlert();
  });

  function resetUploadState() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    fileNameDisplay.textContent = '';
    fileSizeDisplay.textContent = '';

    previewWrapper.classList.add('d-none');
    dropZonePrompt.classList.remove('d-none');

    analyzeBtn.disabled = true;
  }

  // Handle Analyze button click (Prepared for future API integration)
  analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) {
      showError('No File Selected', 'Please select a valid image before starting analysis.');
      return;
    }

    console.log('Image selected and ready for future API integration:', selectedFile.name);
    alert(`File "${selectedFile.name}" selected and ready for future API integration!`);
  });

  // Helper: Format file size
  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Helper: Show error alert
  function showError(title, message) {
    alertTitle.textContent = title;
    alertMessage.textContent = message;
    validationAlert.classList.remove('d-none');
  }

  // Helper: Hide error alert
  function hideAlert() {
    validationAlert.classList.add('d-none');
  }

  dismissAlertBtn.addEventListener('click', () => {
    hideAlert();
  });
});
