document.addEventListener("DOMContentLoaded", () => {
	const lightbox = document.createElement("dialog");
	const lightboxShell = document.createElement("div");
	const lightboxContent = document.createElement("div");
	const lightboxClose = document.createElement("button");

	lightbox.className = "media-lightbox";
	lightboxShell.className = "media-lightbox-shell";
	lightboxContent.className = "media-lightbox-content";
	lightboxClose.className = "media-lightbox-close";
	lightboxClose.type = "button";
	lightboxClose.setAttribute("aria-label", "Close media");
	lightboxClose.textContent = "×";
	lightboxShell.append(lightboxContent, lightboxClose);
	lightbox.appendChild(lightboxShell);
	document.body.appendChild(lightbox);

	const closeLightbox = () => {
		lightbox.close();
		lightboxContent.replaceChildren();
	};

	const openLightbox = (media) => {
		lightboxContent.replaceChildren(media);
		lightbox.showModal();
	};

	lightboxClose.addEventListener("click", closeLightbox);
	lightbox.addEventListener("click", (event) => {
		if (event.target === lightbox || event.target === lightboxShell) {
			closeLightbox();
		}
	});
	lightbox.addEventListener("close", () => lightboxContent.replaceChildren());

	const progress = document.createElement("div");
	const progressFill = document.createElement("span");
	progress.className = "reading-progress";
	progress.setAttribute("aria-hidden", "true");
	progress.appendChild(progressFill);
	document.body.prepend(progress);

	const updateProgress = () => {
		const scrollable = document.documentElement.scrollHeight - window.innerHeight;
		const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
		const percentage = Math.min(100, Math.max(0, ratio * 100));
		progress.style.clipPath = `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
	};

	window.addEventListener("scroll", updateProgress, { passive: true });
	window.addEventListener("resize", updateProgress);
	updateProgress();

	const skippableDetails = document.querySelectorAll("details");
	for (const details of skippableDetails) {
		const summary = details.querySelector(":scope > summary");
		if (summary?.textContent.toLowerCase().includes("feel free to skip")) {
			details.open = false;
		}
	}

	const headings = Array.from(document.querySelectorAll(".page-body h1, .page-body h2"));
	if (headings.length > 0) {
		const outline = document.createElement("nav");
		const outlineTitle = document.createElement("p");
		const outlineLinks = new Map();

		outline.className = "article-outline";
		outline.setAttribute("aria-label", "Article outline");
		outlineTitle.className = "article-outline-title";
		outlineTitle.textContent = "On this page";
		outline.appendChild(outlineTitle);

		for (const [index, heading] of headings.entries()) {
			if (!heading.id) {
				heading.id = `section-${index + 1}`;
			}

			const link = document.createElement("a");
			link.href = `#${heading.id}`;
			link.textContent = heading.textContent.trim();
			link.dataset.level = heading.tagName === "H1" ? "1" : "2";
			outline.appendChild(link);
			outlineLinks.set(heading, link);
		}

		document.body.appendChild(outline);

		const setActiveHeading = () => {
			let activeHeading = headings[0];
			for (const heading of headings) {
				if (heading.getBoundingClientRect().top <= 150) {
					activeHeading = heading;
				} else {
					break;
				}
			}

			for (const [heading, link] of outlineLinks) {
				link.classList.toggle("is-active", heading === activeHeading);
			}
		};

		window.addEventListener("scroll", setActiveHeading, { passive: true });
		window.addEventListener("resize", setActiveHeading);
		setActiveHeading();
	}

	const videoLinks = document.querySelectorAll('figure .source a[href$=".mp4"]');
	const youtubeLinks = document.querySelectorAll(
		'figure .source a[href*="youtube.com/watch"], figure .source a[href*="youtu.be/"]',
	);
	const autoplayTargets = [];

	const initializeVideoSpeed = (video) => {
		video.defaultPlaybackRate = 2;
		video.playbackRate = 2;
	};

	for (const link of videoLinks) {
		const frame = document.createElement("div");
		const video = document.createElement("video");
		const expand = document.createElement("button");
		frame.className = "media-frame";
		video.controls = true;
		video.preload = "metadata";
		video.playsInline = true;
		video.muted = true;
		video.src = link.href;
		video.className = "local-video";
		initializeVideoSpeed(video);
		video.addEventListener("loadedmetadata", () => initializeVideoSpeed(video), { once: true });
		expand.className = "media-expand-button";
		expand.type = "button";
		expand.setAttribute("aria-label", "Open video");
		expand.addEventListener("click", () => {
			const modalVideo = document.createElement("video");
			modalVideo.src = video.src;
			modalVideo.controls = true;
			modalVideo.autoplay = true;
			modalVideo.playsInline = true;
			modalVideo.muted = video.muted;
			initializeVideoSpeed(modalVideo);
			modalVideo.addEventListener("loadedmetadata", () => initializeVideoSpeed(modalVideo), { once: true });
			openLightbox(modalVideo);
		});

		frame.append(video, expand);
		link.parentElement.replaceWith(frame);
		autoplayTargets.push({
			element: frame,
			play: () => {
				video.currentTime = 0;
				void video.play().catch(() => {});
			},
			pause: () => video.pause(),
		});
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
		const expand = document.createElement("button");
		container.className = "youtube-embed";

		const iframe = document.createElement("iframe");
		iframe.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}?enablejsapi=1&playsinline=1&mute=1`;
		iframe.title = "YouTube video player";
		iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
		iframe.allowFullscreen = true;
		iframe.style.width = "100%";
		iframe.style.height = "100%";
		iframe.style.border = "0";
		expand.className = "media-expand-button";
		expand.type = "button";
		expand.setAttribute("aria-label", "Open YouTube video");
		expand.addEventListener("click", () => {
			const modalIframe = iframe.cloneNode();
			modalIframe.src = `${iframe.src}&autoplay=1`;
			modalIframe.allow = iframe.allow;
			modalIframe.allowFullscreen = true;
			modalIframe.addEventListener("load", () => {
				modalIframe.contentWindow?.postMessage(JSON.stringify({
					event: "command",
					func: "setPlaybackRate",
					args: [2],
				}), "*");
			});
			openLightbox(modalIframe);
		});

		container.append(iframe, expand);
		link.parentElement.replaceWith(container);
		const youtubeCommand = (func, args = []) => {
			iframe.contentWindow?.postMessage(JSON.stringify({
				event: "command",
				func,
				args,
			}), "*");
		};
		iframe.addEventListener("load", () => youtubeCommand("setPlaybackRate", [2]));
		autoplayTargets.push({
			element: container,
			play: () => {
				youtubeCommand("seekTo", [0, true]);
				youtubeCommand("setPlaybackRate", [2]);
				youtubeCommand("playVideo");
			},
			pause: () => youtubeCommand("pauseVideo"),
		});
	}

	const mediaObserver = new IntersectionObserver((entries) => {
		for (const entry of entries) {
			const target = autoplayTargets.find(({ element }) => element === entry.target);
			if (!target) {
				continue;
			}

			if (entry.isIntersecting && entry.intersectionRatio >= 0.6) {
				target.play();
			} else {
				target.pause();
			}
		}
	}, { threshold: [0, 0.6] });

	for (const { element } of autoplayTargets) {
		mediaObserver.observe(element);
	}

	const tables = document.querySelectorAll("article table");
	for (const table of tables) {
		if (table.parentElement?.classList.contains("table-scroll")) {
			continue;
		}

		const wrapper = document.createElement("div");
		wrapper.className = "table-scroll";
		table.before(wrapper);
		wrapper.appendChild(table);
	}

	const images = document.querySelectorAll("article figure.image img");
	for (const image of images) {
		const link = image.closest("a");
		const openImage = (event) => {
			event.preventDefault();
			const modalImage = document.createElement("img");
			modalImage.src = link?.href || image.currentSrc || image.src;
			modalImage.alt = image.alt;
			openLightbox(modalImage);
		};

		if (link) {
			link.addEventListener("click", openImage);
		} else {
			image.tabIndex = 0;
			image.setAttribute("role", "button");
			image.addEventListener("click", openImage);
			image.addEventListener("keydown", (event) => {
				if (event.key === "Enter" || event.key === " ") {
					openImage(event);
				}
			});
		}
	}
});
