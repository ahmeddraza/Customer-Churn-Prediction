// ===== DOM Elements =====
const form = document.getElementById('predictionForm');
const featureBars = document.querySelectorAll('.feature-bar');

// ===== Initialize on Page Load =====
document.addEventListener('DOMContentLoaded', function() {
    initializeAnimations();
    initializeFormValidation();
    animateFeatureBars();
    addFormInteractivity();
});

// ===== Animate Feature Bars =====
function animateFeatureBars() {
    // Animate SHAP feature bars
    featureBars.forEach(bar => {
        const width = parseFloat(bar.getAttribute('data-width'));
        // Ensure width is between 0 and 100
        const clampedWidth = Math.min(Math.max(width, 0), 100);
        
        // Delay animation slightly for stagger effect
        setTimeout(() => {
            bar.style.width = clampedWidth + '%';
        }, 100);
    });
}

// ===== Smooth Scroll Animations =====
function initializeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1
    });

    // Observe all result cards
    document.querySelectorAll('.result-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(card);
    });
}

// ===== Form Validation & Enhancement =====
function initializeFormValidation() {
    if (!form) return;

    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        // Add validation on blur
        input.addEventListener('blur', function() {
            validateField(this);
        });

        // Remove error on focus
        input.addEventListener('focus', function() {
            this.classList.remove('error');
            const errorMsg = this.parentElement.querySelector('.error-message');
            if (errorMsg) errorMsg.remove();
        });
    });

    // Form submit with loading state
    form.addEventListener('submit', function(e) {
        let isValid = true;
        
        inputs.forEach(input => {
            if (!validateField(input)) {
                isValid = false;
            }
        });

        if (isValid) {
            showLoadingState();
        } else {
            e.preventDefault();
            scrollToFirstError();
        }
    });
}

// ===== Validate Individual Field =====
function validateField(field) {
    const value = field.value.trim();
    const isRequired = field.hasAttribute('required');
    
    if (isRequired && !value) {
        showError(field, 'This field is required');
        return false;
    }

    if (field.type === 'number') {
        const min = parseFloat(field.getAttribute('min'));
        const max = parseFloat(field.getAttribute('max'));
        const numValue = parseFloat(value);

        if (value && isNaN(numValue)) {
            showError(field, 'Please enter a valid number');
            return false;
        }

        if (!isNaN(min) && numValue < min) {
            showError(field, `Value must be at least ${min}`);
            return false;
        }

        if (!isNaN(max) && numValue > max) {
            showError(field, `Value must be at most ${max}`);
            return false;
        }
    }

    removeError(field);
    return true;
}

// ===== Show Error Message =====
function showError(field, message) {
    field.classList.add('error');
    
    // Remove existing error message
    const existingError = field.parentElement.querySelector('.error-message');
    if (existingError) existingError.remove();
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    errorDiv.style.color = '#ef4444';
    errorDiv.style.fontSize = '0.875rem';
    errorDiv.style.marginTop = '0.25rem';
    field.parentElement.appendChild(errorDiv);
}

// ===== Remove Error Message =====
function removeError(field) {
    field.classList.remove('error');
    const errorMsg = field.parentElement.querySelector('.error-message');
    if (errorMsg) errorMsg.remove();
}

// ===== Scroll to First Error =====
function scrollToFirstError() {
    const firstError = document.querySelector('.error');
    if (firstError) {
        firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        firstError.focus();
    }
}

// ===== Show Loading State =====
function showLoadingState() {
    const submitButton = form.querySelector('.submit-button');
    if (!submitButton) return;

    submitButton.innerHTML = `
        <i class="fas fa-spinner fa-spin"></i>
        <span>Analyzing...</span>
    `;
    submitButton.disabled = true;
    submitButton.style.opacity = '0.7';
    submitButton.style.cursor = 'not-allowed';
}

// ===== Add Form Interactivity =====
function addFormInteractivity() {
    // Auto-calculate total revenue suggestion
    const monthlyCharge = document.getElementById('monthly_charge');
    const tenure = document.getElementById('tenure_in_months');
    const totalRevenue = document.getElementById('total_revenue');

    if (monthlyCharge && tenure && totalRevenue) {
        const updateRevenue = () => {
            const monthly = parseFloat(monthlyCharge.value) || 0;
            const months = parseFloat(tenure.value) || 0;
            if (monthly > 0 && months > 0 && !totalRevenue.value) {
                totalRevenue.value = (monthly * months).toFixed(2);
                totalRevenue.style.background = '#fef3c7';
                setTimeout(() => {
                    totalRevenue.style.background = '';
                }, 1000);
            }
        };

        monthlyCharge.addEventListener('blur', updateRevenue);
        tenure.addEventListener('blur', updateRevenue);
    }

    // Internet service dependencies
    const internetService = document.getElementById('internet_service');
    const internetType = document.getElementById('internet_type');
    const internetRelatedFields = [
        'online_security', 'online_backup', 'device_protection_plan',
        'premium_tech_support', 'streaming_tv', 'streaming_movies',
        'streaming_music', 'unlimited_data'
    ];

    if (internetService) {
        internetService.addEventListener('change', function() {
            if (this.value === 'No') {
                if (internetType) {
                    internetType.value = 'None';
                    internetType.disabled = true;
                }
                internetRelatedFields.forEach(fieldId => {
                    const field = document.getElementById(fieldId);
                    if (field) {
                        field.value = 'No';
                        field.style.opacity = '0.5';
                    }
                });
            } else {
                if (internetType) {
                    internetType.disabled = false;
                }
                internetRelatedFields.forEach(fieldId => {
                    const field = document.getElementById(fieldId);
                    if (field) {
                        field.style.opacity = '1';
                    }
                });
            }
        });
    }

    // Phone service dependencies
    const phoneService = document.getElementById('phone_service');
    const multipleLines = document.getElementById('multiple_lines');

    if (phoneService && multipleLines) {
        phoneService.addEventListener('change', function() {
            if (this.value === 'No') {
                multipleLines.value = 'No';
                multipleLines.disabled = true;
                multipleLines.style.opacity = '0.5';
            } else {
                multipleLines.disabled = false;
                multipleLines.style.opacity = '1';
            }
        });
    }
}

// ===== Number Input Enhancement =====
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('wheel', function(e) {
        e.preventDefault();
    });
});

// ===== Smooth Scroll for Navigation =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ===== Add Tooltip Functionality =====
function addTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function(e) {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.getAttribute('data-tooltip');
            tooltip.style.cssText = `
                position: absolute;
                background: #1e293b;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                font-size: 0.875rem;
                z-index: 1000;
                pointer-events: none;
                white-space: nowrap;
            `;
            
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
            tooltip.style.left = (rect.left + rect.width / 2 - tooltip.offsetWidth / 2) + 'px';
            
            this.tooltipElement = tooltip;
        });
        
        element.addEventListener('mouseleave', function() {
            if (this.tooltipElement) {
                this.tooltipElement.remove();
                this.tooltipElement = null;
            }
        });
    });
}

addTooltips();

// ===== Animate Numbers (Counter Effect) =====
function animateNumber(element, start, end, duration) {
    let startTime = null;
    
    function animation(currentTime) {
        if (!startTime) startTime = currentTime;
        const progress = Math.min((currentTime - startTime) / duration, 1);
        
        const value = start + (end - start) * progress;
        element.textContent = Math.floor(value);
        
        if (progress < 1) {
            requestAnimationFrame(animation);
        }
    }
    
    requestAnimationFrame(animation);
}

// Animate metric values on load
document.querySelectorAll('.metric-value').forEach(element => {
    const text = element.textContent;
    const match = text.match(/[\d,]+\.?\d*/);
    if (match) {
        const value = parseFloat(match[0].replace(',', ''));
        if (!isNaN(value)) {
            element.textContent = '0';
            setTimeout(() => {
                animateNumber(element, 0, value, 1000);
            }, 300);
        }
    }
});

// ===== Print Results =====
function printResults() {
    window.print();
}

// ===== Export to PDF (placeholder) =====
function exportToPDF() {
    alert('PDF export feature coming soon!');
}

// ===== Console Log for Debug =====
console.log('ChurnAI Dashboard Loaded Successfully ✓');
console.log('Dynamic Thresholds: Enabled');
console.log('SHAP Explainability: Active');