const API_BASE = '';
let selectedFile = null;

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const analyzeText = document.getElementById('analyzeText');
const loadingSpinner = document.getElementById('loadingSpinner');

// File input change handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop handlers
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
        handleFileSelect(file);
    } else {
        alert('Vui lòng chọn file ảnh hợp lệ!');
    }
});

// Handle file selection
function handleFileSelect(file) {
    selectedFile = file;
    
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

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) {
        alert('Vui lòng chọn ảnh trước!');
        return;
    }
    
    // Show loading state
    const analyzeBtn = event.target;
    analyzeBtn.disabled = true;
    analyzeText.textContent = 'Đang phân tích...';
    loadingSpinner.style.display = 'inline-block';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
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
        
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        console.error('Error stack:', error.stack);
        displayError('Đã xảy ra lỗi khi phân tích ảnh. Vui lòng thử lại!');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeText.textContent = 'Phân tích';
        loadingSpinner.style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = '';
    
    if (!data.success) {
        resultsContainer.innerHTML = `
            <div class="no-animals-message">
                <h3>⚠️ ${data.message}</h3>
                <p>Không phát hiện động vật nào trong ảnh. Vui lòng thử ảnh khác!</p>
            </div>
        `;
        return;
    }
    
    data.objects.forEach((obj, index) => {
        const objectCard = createObjectCard(obj, index);
        resultsContainer.appendChild(objectCard);
    });
    
    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create object card
function createObjectCard(obj, index) {
    const card = document.createElement('div');
    card.className = 'object-card';
    
    const confidencePercent = (obj.confidence * 100).toFixed(1);
    
    card.innerHTML = `
        <div class="object-header">
            <div class="object-title">🐾 Động vật ${index + 1}: ${obj.predicted_class}</div>
            <div class="confidence-badge">Độ tin cậy: ${confidencePercent}%</div>
        </div>
        
        ${obj.cropped_image ? `
            <div class="cropped-image-section">
                <h3>Ảnh đối tượng đã cắt</h3>
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
function displayError(message) {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = `
        <div class="error-message">
            <h3>❌ Lỗi</h3>
            <p>${message}</p>
        </div>
    `;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API not available:', error);
    }
});
