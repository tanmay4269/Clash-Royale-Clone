document.addEventListener("DOMContentLoaded", () => {
	const videoLinks = document.querySelectorAll('figure .source a[href$=".mp4"]');

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
});
