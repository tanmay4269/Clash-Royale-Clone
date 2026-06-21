document.addEventListener("DOMContentLoaded", () => {
	const revisions = [
		{
			id: "386aa1bf-7b79-808d-8071-c7522955bc8b",
			context: "Additional changes that can be made",
			html: `
				<h4>Highest priority: evaluation and observability</h4>
				<ul>
					<li>Run a frozen common round-robin across final checkpoints, with several seeds and swapped sides.</li>
					<li>Log reward components, value and return histograms, advantage sign ratio, non-forced skip rate, per-head gradient norms and checkpoint sampling statistics.</li>
					<li>Add win-conditioned and loss-conditioned deployment heatmaps, plus win rate by opening action.</li>
				</ul>
				<h4>Simulator fidelity</h4>
				<ul>
					<li>Implement tower-dependent territory unlock and separate troop and spell deployment masks.</li>
					<li>Fix bridge-edge and wall pathing, sudden-death correctness, giant placement and true flying behavior for minions.</li>
					<li>Ablate custom overtime and asymmetric tower settings before presenting them as helpful inductive biases.</li>
				</ul>
				<h4>Network and learning</h4>
				<ul>
					<li>Replace masked-mean-only aggregation with mean plus max or learned set pooling.</li>
					<li>Add a full feed-forward sublayer to Deep Sets attention and test multiple attention blocks.</li>
					<li>Compare an explicit spatial feature-map encoder against entity-only encoders.</li>
					<li>Test short temporal context, then opponent-action prediction as an auxiliary objective.</li>
					<li>Use a curriculum from scripted to random to checkpoint opponents instead of jumping directly into unstable self-play.</li>
				</ul>
				<h4>Engineering and presentation</h4>
				<ul>
					<li>Make <code>game.py</code> reuse simulator and RL utilities instead of duplicating behavior.</li>
					<li>Show both players’ Elo in recorded games and preserve diagnostic state across resumes.</li>
					<li>Rename and package the repository around the RL-playground goal so the project is discoverable without implying full Clash Royale fidelity.</li>
				</ul>
			`,
		},
	];

	const entries = revisions
		.map((revision) => ({
			revision,
			heading: document.getElementById(revision.id),
		}))
		.filter(({ heading }) => heading);

	for (const { revision, heading } of entries) {
		const block = document.createElement("aside");
		block.className = "revision-block";
		block.setAttribute("aria-label", `Revision guidance for ${revision.context}`);
		block.innerHTML = `
			<div class="revision-block-header">
				<p class="revision-block-label">Revision guidance</p>
				<p class="revision-block-context">${revision.context}</p>
			</div>
			${revision.html}
		`;

		heading.parentElement?.after(block);
	}
});
