document.addEventListener("DOMContentLoaded", () => {
	const videoLinks = document.querySelectorAll('figure .source a[href$=".mp4"]');
	const youtubeLinks = document.querySelectorAll(
		'figure .source a[href*="youtube.com/watch"], figure .source a[href*="youtu.be/"]',
	);

	for (const link of videoLinks) {
		const video = document.createElement("video");
		video.controls = true;
		video.preload = "metadata";
		video.playsInline = true;
		video.src = link.href;
		video.style.display = "block";
		video.style.width = "100%";
		video.style.maxWidth = "720px";
		video.style.margin = "0 auto";

		link.parentElement.replaceWith(video);
	}

	for (const link of youtubeLinks) {
		const url = new URL(link.href);
		const videoId = url.hostname === "youtu.be"
			? url.pathname.slice(1)
			: url.searchParams.get("v");

		if (!videoId) {
			continue;
		}

		const container = document.createElement("div");
		container.style.position = "relative";
		container.style.width = "100%";
		container.style.maxWidth = "720px";
		container.style.aspectRatio = "16 / 9";
		container.style.margin = "0 auto";

		const iframe = document.createElement("iframe");
		iframe.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}`;
		iframe.title = "YouTube video player";
		iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
		iframe.allowFullscreen = true;
		iframe.style.width = "100%";
		iframe.style.height = "100%";
		iframe.style.border = "0";

		container.appendChild(iframe);
		link.parentElement.replaceWith(container);
	}
});
