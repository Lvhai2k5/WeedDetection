$(document).ready(function () {
    // Sidebar Toggle
    $('#sidebarCollapse').on('click', function () {
        $('#sidebar').toggleClass('active');
    });

    // Logger Utility
    function addLog(message, type = 'info') {
        const time = new Date().toLocaleTimeString();
        let cssClass = 'text-light';
        if (type === 'error') cssClass = 'log-error';
        if (type === 'success') cssClass = 'log-success';
        if (type === 'warn') cssClass = 'log-warn';
        
        const logEntry = `<div class="${cssClass}">[${time}] ${message}</div>`;
        const $container = $('#log-container');
        $container.append(logEntry);
        $container.scrollTop($container[0].scrollHeight);
    }

    $('#btn-clear-logs').on('click', function() {
        $('#log-container').empty();
        addLog('Logs cleared.', 'info');
    });

    // ESP32 Connection
    $('#btn-connect').on('click', function () {
        const ip = $('#esp-ip').val().trim();
        if (!ip) {
            alert('Vui lòng nhập IP ESP32');
            return;
        }

        const $btn = $(this);
        $btn.prop('disabled', true).html('<i class="fa-solid fa-spinner fa-spin"></i>');
        addLog(`Đang kết nối ESP32 tại ${ip}...`, 'info');

        $.ajax({
            url: '/api/esp32/connect-esp32',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ ip: ip }),
            success: function (res) {
                if (res.success) {
                    $('#esp-status-badge')
                        .removeClass('bg-danger')
                        .addClass('bg-success')
                        .text('ESP32: Connected');
                    addLog(res.message, 'success');
                }
            },
            error: function (xhr) {
                const msg = xhr.responseJSON ? xhr.responseJSON.message : 'Connection failed';
                $('#esp-status-badge')
                    .removeClass('bg-success')
                    .addClass('bg-danger')
                    .text('ESP32: Disconnected');
                addLog(msg, 'error');
            },
            complete: function() {
                $btn.prop('disabled', false).text('Connect');
            }
        });
    });

    // Update Dashboard UI with results
    function updateDashboard(data) {
        if (data.blur_warning) {
            $('#blur-warning').removeClass('d-none');
        } else {
            $('#blur-warning').addClass('d-none');
        }

        // Stats
        $('#stat-density').text(`${data.density.toFixed(2)}%`);
        $('#stat-density-level').text(`Level: ${data.density_level}`);
        $('#stat-young').text(`${data.young_density.toFixed(2)}%`);
        $('#stat-mature').text(`${data.mature_density.toFixed(2)}%`);
        $('#stat-count').text(data.weed_count);
        $('#stat-spray').html(`${data.spray_ms} <small class="fs-6">ms</small>`);

        // Images
        if (data.images) {
            if(data.images.yolo) $('#img-yolo').attr('src', data.images.yolo);
            if(data.images.mask) $('#img-mask').attr('src', data.images.mask);
            if(data.images.young) $('#img-young').attr('src', data.images.young);
            if(data.images.mature) $('#img-mature').attr('src', data.images.mature);
        }

        // Logs
        if (data.logs && data.logs.length > 0) {
            data.logs.forEach(log => {
                let type = 'info';
                if (log.includes('⚠') || log.includes('LỖI')) type = 'warn';
                if (log.includes('✅')) type = 'success';
                addLog(log, type);
            });
        }
    }

    // Capture & Detect
    $('#btn-capture-detect').on('click', function() {
        if ($('#esp-status-badge').hasClass('bg-danger')) {
            alert('Chưa kết nối ESP32!');
            return;
        }

        const $btn = $(this);
        $btn.prop('disabled', true).html('<i class="fa-solid fa-spinner fa-spin"></i> Processing...');
        addLog('Đang yêu cầu ảnh từ ESP32 và xử lý YOLO...', 'info');

        $.ajax({
            url: '/api/detection/capture-detect',
            type: 'POST',
            success: function(res) {
                if(res.success) {
                    updateDashboard(res);
                } else {
                    addLog(res.message, 'error');
                }
            },
            error: function(xhr) {
                addLog('Lỗi server khi capture ảnh.', 'error');
            },
            complete: function() {
                $btn.prop('disabled', false).html('<i class="fa-solid fa-camera"></i> Capture & Detect');
            }
        });
    });

    // Upload Local Image
    $('#file-upload').on('change', function() {
        const file = this.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        addLog(`Đang tải lên và phân tích ảnh: ${file.name}...`, 'info');

        $.ajax({
            url: '/api/detection/upload-image',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(res) {
                if(res.success) {
                    updateDashboard(res);
                } else {
                    addLog(res.message, 'error');
                }
            },
            error: function() {
                addLog('Lỗi khi upload ảnh.', 'error');
            },
            complete: function() {
                $('#file-upload').val(''); // reset
            }
        });
    });

    // Video Stream
    let streamActive = false;
    $('#btn-start-stream').on('click', function() {
        if ($('#esp-status-badge').hasClass('bg-danger')) {
            alert('Chưa kết nối ESP32!');
            return;
        }
        if (streamActive) return;

        streamActive = true;
        addLog('Bắt đầu xem camera trực tiếp...', 'info');
        // Set main image to stream
        $('#img-yolo').attr('src', '/api/detection/video-stream?' + new Date().getTime());
    });

    $('#btn-stop-stream').on('click', function() {
        if (!streamActive) return;
        streamActive = false;
        addLog('Đã dừng xem camera.', 'warn');
        // Stop stream by replacing src with empty or a placeholder
        $('#img-yolo').attr('src', '');
    });

    // History Modal
    const historyModal = new bootstrap.Modal(document.getElementById('historyModal'));
    $('#nav-history').on('click', function(e) {
        e.preventDefault();
        
        // Fetch history
        $.ajax({
            url: '/api/detection/history',
            type: 'GET',
            success: function(res) {
                if(res.success) {
                    const tbody = $('#history-table tbody');
                    tbody.empty();
                    
                    if(res.data.length === 0) {
                        tbody.append('<tr><td colspan="7" class="text-center">No history found</td></tr>');
                    } else {
                        res.data.forEach(item => {
                            // Format date
                            const date = new Date(item.created_at).toLocaleString();
                            const imgHtml = item.image_path ? 
                                `<a href="/static/${item.image_path}" target="_blank">View Image</a>` : 
                                'N/A';
                                
                            tbody.append(`
                                <tr>
                                    <td>${item.id}</td>
                                    <td>${imgHtml}</td>
                                    <td>${date}</td>
                                    <td>${item.weed_density.toFixed(2)}%</td>
                                    <td>${item.weed_count}</td>
                                    <td>${item.spray_time}</td>
                                    <td>${item.blur_score.toFixed(1)}</td>
                                </tr>
                            `);
                        });
                    }
                    historyModal.show();
                }
            }
        });
    });
});
