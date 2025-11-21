const API_BASE = '';
let selectedFiles = {
    1: null,
    2: null
};

// DOM Elements - will be initialized after DOM loads
let uploadBox1, uploadBox2, fileInput1, fileInput2;
let previewSection1, previewSection2, previewImage1, previewImage2;
let resultsSection, resultsContainer1, resultsContainer2;
let analyzeText, loadingSpinner, analyzeBtn;

// Initialize DOM elements and event listeners after page loads
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM Elements
    uploadBox1 = document.getElementById('uploadBox1');
    uploadBox2 = document.getElementById('uploadBox2');
    fileInput1 = document.getElementById('fileInput1');
    fileInput2 = document.getElementById('fileInput2');
    previewSection1 = document.getElementById('previewSection1');
    previewSection2 = document.getElementById('previewSection2');
    previewImage1 = document.getElementById('previewImage1');
    previewImage2 = document.getElementById('previewImage2');
    resultsSection = document.getElementById('resultsSection');
    resultsContainer1 = document.getElementById('resultsContainer1');
    resultsContainer2 = document.getElementById('resultsContainer2');
    analyzeText = document.getElementById('analyzeText');
    loadingSpinner = document.getElementById('loadingSpinner');
    analyzeBtn = document.getElementById('analyzeBtn');

    // File input change handlers
    fileInput1.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file, 1);
        }
    });

    fileInput2.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file, 2);
        }
    });

    // Setup drag and drop for both upload boxes
    setupDragAndDrop(uploadBox1, 1);
    setupDragAndDrop(uploadBox2, 2);

    // Check API health
    checkAPIHealth();
});

// Setup drag and drop for both upload boxes
function setupDragAndDrop(uploadBox, imageNumber) {
    if (!uploadBox) return;
    
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file, imageNumber);
        } else {
            alert('Vui lòng chọn file ảnh hợp lệ!');
        }
    });
}

// Handle file selection
function handleFileSelect(file, imageNumber) {
    selectedFiles[imageNumber] = file;
    
    const uploadBox = document.getElementById(`uploadBox${imageNumber}`);
    const previewSection = document.getElementById(`previewSection${imageNumber}`);
    const previewImage = document.getElementById(`previewImage${imageNumber}`);
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Reset upload for specific image
function resetUpload(imageNumber) {
    selectedFiles[imageNumber] = null;
    
    const fileInput = document.getElementById(`fileInput${imageNumber}`);
    const uploadBox = document.getElementById(`uploadBox${imageNumber}`);
    const previewSection = document.getElementById(`previewSection${imageNumber}`);
    const resultsContainer = document.getElementById(`resultsContainer${imageNumber}`);
    
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    
    if (resultsContainer) {
        resultsContainer.innerHTML = '';
    }
}

// Reset all uploads
function resetAll() {
    resetUpload(1);
    resetUpload(2);
    resultsSection.style.display = 'none';
}

// Analyze images
async function analyzeImages() {
    if (!selectedFiles[1]) {
        alert('Vui lòng chọn ít nhất ảnh thứ nhất!');
        return;
    }
    
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeText.textContent = 'Đang phân tích...';
    loadingSpinner.style.display = 'inline-block';
    
    try {
        // Clear previous results
        resultsContainer1.innerHTML = '';
        resultsContainer2.innerHTML = '';
        resultsSection.style.display = 'block';
        
        // Analyze image 1
        if (selectedFiles[1]) {
            resultsContainer1.innerHTML = '<div class="loading-message">⏳ Đang xử lý ảnh 1...</div>';
            const data1 = await analyzeSingleImage(selectedFiles[1]);
            displayResults(data1, 1);
        }
        
        // Analyze image 2 if provided
        if (selectedFiles[2]) {
            resultsContainer2.innerHTML = '<div class="loading-message">⏳ Đang xử lý ảnh 2...</div>';
            const data2 = await analyzeSingleImage(selectedFiles[2]);
            displayResults(data2, 2);
        } else {
            resultsContainer2.innerHTML = '<div class="no-image-message">💡 Không có ảnh thứ hai để phân tích</div>';
        }
        
        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error('Error:', error);
        displayError('Đã xảy ra lỗi khi phân tích ảnh. Vui lòng thử lại!', 1);
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeText.textContent = 'Phân tích';
        loadingSpinner.style.display = 'none';
    }
}

// Analyze single image
async function analyzeSingleImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
    });
    
    console.log('Response status:', response.status);
    
    if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const responseText = await response.text();
    console.log('Response size:', responseText.length, 'chars');
    
    let data;
    try {
        data = JSON.parse(responseText);
        console.log('Parsed data:', data);
    } catch (parseError) {
        console.error('JSON parse error:', parseError);
        console.error('Response text preview:', responseText.substring(0, 500));
        throw new Error('Failed to parse server response');
    }
    
    return data;
}

// Display results
function displayResults(data, imageNumber) {
    const resultsContainer = document.getElementById(`resultsContainer${imageNumber}`);
    resultsContainer.innerHTML = '';
    
    if (!data.success) {
        resultsContainer.innerHTML = `
            <div class="no-animals-message">
                <h3>⚠️ ${data.message}</h3>
                <p>Không phát hiện động vật nào trong ảnh.</p>
            </div>
        `;
        return;
    }
    
    // Add image header
    const header = document.createElement('div');
    header.className = 'result-header';
    header.innerHTML = `<h3>📸 Kết quả ảnh ${imageNumber}</h3>`;
    resultsContainer.appendChild(header);
    
    data.objects.forEach((obj, index) => {
        const objectCard = createObjectCard(obj, index);
        resultsContainer.appendChild(objectCard);
    });
}

// Create object card
function createObjectCard(obj, index) {
    const card = document.createElement('div');
    card.className = 'object-card';
    
    const confidencePercent = (obj.confidence * 100).toFixed(1);
    
    card.innerHTML = `
        <div class="object-header">
            <div class="object-title">🐾 Động vật ${index + 1}: ${obj.predicted_class}</div>
        </div>
        
        ${obj.cropped_image ? `
            <div class="cropped-image-section">
                <h3>Ảnh đối tượng</h3>
                <img src="data:image/jpeg;base64,${obj.cropped_image}" alt="Cropped animal" class="cropped-image">
            </div>
        ` : ''}
        
        <div class="summary-section">
            <h3>Thông tin chi tiết</h3>
            <p class="summary-text">${obj.summary}</p>
        </div>
        
        <div class="similar-images-section">
            <h3>🔍 Ảnh tương tự (Top ${obj.similar_images.length})</h3>
            <div class="similar-images-grid">
                ${obj.similar_images.map(img => createSimilarImageCard(img)).join('')}
            </div>
        </div>
    `;
    
    return card;
}

// Create similar image card
function createSimilarImageCard(img) {
    const score = (1 - img.score).toFixed(3); // Convert distance to similarity
    const scorePercent = ((1 - img.score) * 100).toFixed(1);
    
    // Extract image path from metadata
    let imagePath = img.metadata;
    if (imagePath.startsWith('data/')) {
        imagePath = `/${imagePath}`;
    }
    
    return `
        <div class="similar-image-card">
            <img src="${imagePath}" alt="Similar image" class="similar-image" 
                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22150%22 height=%22150%22%3E%3Crect fill=%22%23ddd%22 width=%22150%22 height=%22150%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%23999%22%3ENo Image%3C/text%3E%3C/svg%3E'">
            <div class="similar-image-info">
                <span class="similarity-score">${scorePercent}%</span>
            </div>
        </div>
    `;
}

// Display error
function displayError(message, imageNumber) {
    resultsSection.style.display = 'block';
    const resultsContainer = document.getElementById(`resultsContainer${imageNumber || 1}`);
    resultsContainer.innerHTML = `
        <div class="error-message">
            <h3>❌ Lỗi</h3>
            <p>${message}</p>
        </div>
    `;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API not available:', error);
    }
}
