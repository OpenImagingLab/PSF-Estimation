window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    // 初始化所有图片对比滑块
    const containers = document.querySelectorAll('.bal-container');
    
    containers.forEach(container => {
        const handle = container.querySelector('.bal-handle');
        const beforeDiv = container.querySelector('.bal-before');
        let isDragging = false;

        // 鼠标按下事件
        handle.addEventListener('mousedown', function(e) {
            isDragging = true;
            e.preventDefault();
        });

        // 鼠标移动事件
        document.addEventListener('mousemove', function(e) {
            if (!isDragging) return;
            
            const rect = container.getBoundingClientRect();
            const x = Math.min(Math.max(0, e.pageX - rect.left), rect.width);
            const percent = (x / rect.width) * 100;
            
            handle.style.left = `${percent}%`;
            beforeDiv.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
        });

        // 鼠标松开事件
        document.addEventListener('mouseup', function() {
            isDragging = false;
        });

        // 触摸事件支持
        handle.addEventListener('touchstart', function(e) {
            isDragging = true;
            e.preventDefault();
        });

        document.addEventListener('touchmove', function(e) {
            if (!isDragging) return;
            
            const touch = e.touches[0];
            const rect = container.getBoundingClientRect();
            const x = Math.min(Math.max(0, touch.pageX - rect.left), rect.width);
            const percent = (x / rect.width) * 100;
            
            handle.style.left = `${percent}%`;
            beforeDiv.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
        });

        document.addEventListener('touchend', function() {
            isDragging = false;
        });
    });

    document.addEventListener('DOMContentLoaded', function() {
        const containers = document.querySelectorAll('.bal-container-small');
        
        containers.forEach(container => {
            const handle = container.querySelector('.bal-handle');
            const beforeDiv = container.querySelector('.bal-before');
            let isDragging = false;

            // 鼠标按下事件
            handle.addEventListener('mousedown', function(e) {
                isDragging = true;
                e.preventDefault();
            });

            // 鼠标移动事件
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                
                const rect = container.getBoundingClientRect();
                const x = Math.min(Math.max(0, e.pageX - rect.left), rect.width);
                const percent = (x / rect.width) * 100;
                
                handle.style.left = `${percent}%`;
                beforeDiv.style.width = `${percent}%`;
            });

            // 鼠标松开事件
            document.addEventListener('mouseup', function() {
                isDragging = false;
            });

            // 触摸事件支持
            handle.addEventListener('touchstart', function(e) {
                isDragging = true;
                e.preventDefault();
            });

            document.addEventListener('touchmove', function(e) {
                if (!isDragging) return;
                
                const touch = e.touches[0];
                const rect = container.getBoundingClientRect();
                const x = Math.min(Math.max(0, touch.pageX - rect.left), rect.width);
                const percent = (x / rect.width) * 100;
                
                handle.style.left = `${percent}%`;
                beforeDiv.style.width = `${percent}%`;
            });

            document.addEventListener('touchend', function() {
                isDragging = false;
            });
        });
    });

})
