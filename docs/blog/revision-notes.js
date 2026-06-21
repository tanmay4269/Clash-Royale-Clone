document.addEventListener("DOMContentLoaded", () => {
	const revisions = [
		{
			id: "371aa1bf-7b79-80c7-9c06-c3ae5630cdc2",
			context: "The Algorithm",
			html: `
				<h4>Clarify the “from scratch” claim</h4>
				<p>Say that PPO, GAE, rollout storage, checkpoint matchmaking and the model architectures were implemented directly in PyTorch. The project still depends on Gymnasium for the environment interface, Pygame for rendering and Weights & Biases for logging. The meaningful distinction is “no external RL training framework,” not “no external framework.”</p>
				<h4>My suggested addition</h4>
				<p>Start this section with one frozen final-configuration table. The run history changed gamma, minibatch size, entropy coefficient, architecture flags and opponent mode repeatedly. Readers need to know which values describe the final system and which belong only to historical experiments.</p>
			`,
		},
		{
			id: "371aa1bf-7b79-806b-b5cf-e80fc0b7a22e",
			context: "Network",
			html: `
				<h4>Correct the Deep Sets description</h4>
				<ul>
					<li>Use entity shape <code>(B, N, 26)</code>. Deployed units are aggregated with a masked mean, not a sum.</li>
					<li>The current attention option is a single <code>MultiheadAttention</code> layer with residual connection and LayerNorm. It is not yet a full Transformer block because it has no feed-forward sublayer.</li>
					<li>Hand and next-card entities enter the attention sequence and are also flattened into the trunk. Mention this duplicated information path as a current limitation.</li>
					<li>The 418-dimensional trunk is correct for 32-dimensional embeddings: two scalars, six tower embeddings, two pooled battlefield embeddings, four hand embeddings and one next-card embedding.</li>
					<li>The current position head can condition on the chosen card embedding, not merely a one-hot deck index. The CNN decoder then outputs the 18 by 32 position grid.</li>
					<li>Document learned temperature per action head, elixir masking, forced skip, conditional entropy and log probability, orthogonal initialization, actor-head standard deviation <code>0.01</code> and critic-head standard deviation <code>1.0</code>.</li>
				</ul>
				<h4>Explain the design choices requested by the comments</h4>
				<ul>
					<li><strong>Disjoint actor and critic:</strong> prevents value-loss gradients from directly changing the policy representation, at the cost of nearly doubling encoder and trunk compute.</li>
					<li><strong>Tanh and LayerNorm:</strong> were chosen for stable bounded activations, but the article should label this as a design choice unless an isolated activation ablation is shown.</li>
					<li><strong>Pointer decoder:</strong> scores a learned skip token and the four hand embeddings against a state-derived query. This ties card choice to card content instead of hand slot identity.</li>
					<li><strong>Conditional position decoder:</strong> the chosen hand embedding is concatenated with the actor trunk before predicting location. This is a stronger dependency than independently predicting card and position.</li>
				</ul>
				<h4>Correct the Transformer description</h4>
				<p>Meta features are projected and added to the CLS token. The sequence uses four segment types: CLS, towers, deployed entities and hand plus next card. The current encoder uses pre-LayerNorm, zero dropout and a two-layer post-Transformer MLP. These are newer than the prose and diagram comments.</p>
				<h4>Explain run 25 versus run 26 exactly</h4>
				<p>Run 25 used the legacy Deep Sets baseline: one shared encoder and trunk, ReLU, default linear initialization, linear position logits and no card-conditioned position head. Run 26 used the newer Deep Sets bundle: disjoint actor and critic, Tanh, orthogonal initialization with policy and value head scales, a transposed-convolution position decoder, and card-conditioned position prediction. The run did not isolate those factors, so attribute the gain to the bundle, then ablate each component separately.</p>
				<h4>My suggested additions</h4>
				<ul>
					<li>Regenerate both diagrams from the current code and constrain label widths so text does not spill across lines.</li>
					<li>Test mean plus max pooling or learned pooling because masked mean is the largest information bottleneck.</li>
					<li>Turn Deep Sets attention into a complete block with attention, feed-forward network, residuals and LayerNorm, then test one versus two blocks.</li>
					<li>Add a spatial feature-map baseline. The model predicts an 18 by 32 output grid but currently reconstructs spatial structure entirely from entity coordinates.</li>
				</ul>
			`,
		},
		{
			id: "382aa1bf-7b79-8084-9fe0-d2e6c7a84967",
			context: "Training Algorithm",
			html: `
				<h4>Turn the placeholder list into a data-flow explanation</h4>
				<ol>
					<li>Collect independent trajectories in parallel environments. Each environment keeps a separate contiguous buffer so GAE does not cross episode or environment boundaries.</li>
					<li>Compute GAE per environment, then merge buffers and shuffle minibatches for PPO updates.</li>
					<li>Set rollout size as <code>game_duration × 4 FPS × num_games_in_buffer</code>. Explain the tradeoff: larger rollouts give more diverse targets but make each update less frequent and can waste early low-quality experience.</li>
					<li>Use Adam because the gradients are noisy and differently scaled across actor, critic and position heads. Avoid claiming it is optimal without an optimizer ablation.</li>
					<li>Describe the LR finder, linear LR decay and entropy-coefficient schedule separately. The current default entropy coefficient is constant at <code>0.01</code> because initial and final values are equal.</li>
					<li>Describe PPO clipping, optional KL early stopping, global or per-head gradient clipping, minibatch advantage normalization and optional observation/value normalization.</li>
				</ol>
				<h4>Explain self-play matchmaking precisely</h4>
				<p>A checkpoint is saved after at least 100 games when the recent mean score reaches 0.55. During sampling, half the time the latest available checkpoint is chosen. Otherwise, opponents are weighted toward an expected 50 percent matchup using Elo. Each parallel environment owns a separate opponent instance, and both the learner and checkpoint Elo values are updated after games.</p>
				<h4>Separate defaults from historical run settings</h4>
				<p>The current code defaults include gamma <code>0.99</code>, lambda <code>0.95</code>, 40 PPO epochs, minibatch size 2,048 and optional KL early stopping disabled. Historical successful runs used different values, including gamma <code>0.997</code> and minibatch size 256. Put those in the experiment table rather than describing them as current defaults.</p>
				<h4>My suggested additions</h4>
				<ul>
					<li>Include pseudocode for one complete update, from rollout to checkpoint update.</li>
					<li>Log checkpoint-manager score and checkpoint sampling frequencies. These are still missing from the current diagnostics.</li>
					<li>Persist running diagnostic state on resume, not only model, optimizer, Elo and checkpoint metadata.</li>
				</ul>
			`,
		},
		{
			id: "371aa1bf-7b79-8012-bc27-f64d8a45a5ec",
			context: "Experiments",
			html: `
				<h4>Replace the TODO list with an evaluation protocol</h4>
				<p>Training Elo values from separate runs are not directly comparable because each run creates its own opponent population. For the final comparison, freeze representative checkpoints from every architecture and evaluate them in one shared round-robin pool with equal games, swapped player sides and multiple seeds. Report the game matrix, win rates and a rating fitted from that common matrix.</p>
				<h4>Explain DiagnosticsLogger by decision, not by field name</h4>
				<ul>
					<li><strong>Is behavior improving:</strong> score, return, Elo, tower kills and evaluation against fixed scripted and random bots.</li>
					<li><strong>Is the policy collapsing:</strong> skip ratio, card histogram, deployment heatmap and per-head entropy.</li>
					<li><strong>Is PPO using the rollout:</strong> ratio mean, clip fraction, approximate KL and KL per action head.</li>
					<li><strong>Can the critic learn:</strong> critic loss and explained variance. Add value and return histograms because the current logger does not show whether either distribution has collapsed.</li>
					<li><strong>Does the environment terminate meaningfully:</strong> terminated versus truncated games, episode duration and tower kills.</li>
				</ul>
				<p>Add the missing reward decomposition: HP-delta reward, tower-destruction reward, outcome reward and step penalty. Without this, the article cannot establish which reward term drives a result.</p>
				<h4>Explain the BotNet ladder</h4>
				<p>Use fixed references in increasing difficulty: always skip, scripted alternating lane, uniform random, then frozen learned checkpoints. “Solved” should mean a predeclared win-rate threshold across several seeds, not one favorable training curve.</p>
				<h4>My suggested addition</h4>
				<p>Use one summary table with columns for run, opponent, cards, architecture, isolated change, seeds, evaluation games and result. This will prevent the chronology from mixing causal ablations with exploratory runs.</p>
			`,
		},
		{
			id: "382aa1bf-7b79-80db-b27d-ed807ec9c0cd",
			context: "Ablations",
			html: `
				<h4>Use a short, high-leverage ablation matrix</h4>
				<ul>
					<li><strong>Simulator fidelity:</strong> symmetric towers, standard versus custom overtime, dynamic territory unlock, spell-specific targeting and corrected bridge pathing.</li>
					<li><strong>Reward:</strong> outcome only, plus HP damage, plus destruction reward and plus step penalty. Log reward components and termination rate.</li>
					<li><strong>Network:</strong> shared versus disjoint actor-critic, ReLU versus Tanh, linear versus CNN position head, unconditional versus card-conditioned position head, pointer decoder, entity attention and Transformer.</li>
					<li><strong>Training:</strong> gamma 0.99 versus 0.997, minibatch 2,048 versus 256, KL early stopping, forced-skip removal and rollout games per update.</li>
				</ul>
				<h4>Existing evidence to retain, with calibrated claims</h4>
				<ul>
					<li>Gamma 0.997 substantially improved the scripted-bot setting in run 12.</li>
					<li>The step-penalty results conflict across runs 13 and 19, so no conclusion is established.</li>
					<li>KL early stopping and 120 maximum epochs underperformed run 16 in runs 17 and 21.</li>
					<li>Running advantage normalization collapsed behavior in run 18.</li>
					<li>A 100-game rollout buffer performed poorly in run 20.</li>
					<li>Minibatch size 256 improved the run-16 family in run 22.</li>
					<li>Elixir-based card masking and forced skip produced the largest clean behavioral jump in run 16.</li>
				</ul>
				<h4>Protocol</h4>
				<p>Use matched seeds, identical environment versions, equal environment steps and frozen evaluation opponents. Report mean and uncertainty across seeds. For architecture comparisons, also match parameter count or report it explicitly.</p>
				<h4>My suggested addition</h4>
				<p>Prioritize ablations that challenge the article’s strongest claims. In particular, decompose the run-25 to run-26 architecture bundle and directly test whether the visually engaging Deep Sets policy is actually stronger than the Transformer in head-to-head games.</p>
			`,
		},
		{
			id: "371aa1bf-7b79-804d-8f69-eeb3d77ee85f",
			context: "Appendix",
			html: `
				<h4>Restructure the appendix</h4>
				<p>Keep derivations and implementation details here, but move the experiment chronology into the main Experiments section. The appendix should support the argument rather than contain the evidence required to understand the result.</p>
				<h4>My suggested addition</h4>
				<p>Add a notation table that maps mathematical symbols to implementation names such as <code>gae_gamma</code>, <code>gae_lambda</code>, <code>ppo_clip</code>, <code>critic_loss_coef</code> and <code>entropy_loss_coef</code>.</p>
			`,
		},
		{
			id: "371aa1bf-7b79-80a3-8b7c-f4325c33ae74",
			context: "PPO: Proximal Policy Optimisation",
			html: `
				<h4>Correct the problem formulation</h4>
				<p>The agent does not observe the opponent hand or elixir, so the learner operates under partial observability. The underlying simulator may be Markov, but the policy input is not a sufficient statistic for the full state. Call this a partially observable MDP and describe the current feed-forward policy as reactive.</p>
				<h4>Clarify the objective</h4>
				<p>Keep discounting consistent when defining the return and objective. The trajectory objective should not switch from discounted return to an undiscounted sum without explanation.</p>
				<h4>My suggested addition</h4>
				<p>Briefly connect partial observability to the future-work proposals for frame stacking, recurrent memory and opponent modeling.</p>
			`,
		},
		{
			id: "371aa1bf-7b79-809e-82cb-f304083f8039",
			context: "Actor Critic Algorithm",
			html: `
				<h4>Align the derivation with the implementation</h4>
				<p>The current critic is trained against GAE-derived returns using mean squared error. Avoid writing that the critic simply minimizes the square of a one-step advantage unless you explicitly introduce that as a simplified derivation. Also state that actor advantages are detached before the policy update.</p>
				<h4>My suggested addition</h4>
				<p>Explain why actor and critic are disjoint in the implemented default even though textbook actor-critic diagrams often show a shared backbone.</p>
			`,
		},
		{
			id: "371aa1bf-7b79-80c1-a3b0-c79218b80e66",
			context: "PPO Algorithm",
			html: `
				<h4>Add implementation-specific details</h4>
				<ul>
					<li>The joint action log probability is skip log probability plus card and position log probabilities only when the action is not skip.</li>
					<li>Entropy follows the same conditional structure, preventing irrelevant card and position entropy from dominating skip steps.</li>
					<li>Advantages are normalized per minibatch. Returns can optionally be normalized, and observation normalization is separately configurable.</li>
					<li>The implementation uses policy clipping and an unclipped value MSE. Optional KL early stopping is an additional safety mechanism, not part of the base clipped objective.</li>
				</ul>
				<h4>Correct the optimizer explanation</h4>
				<p>Random minibatches reduce correlation, but Adam and SGD do not require perfectly IID samples. The important practical issue is that GAE must be computed on ordered trajectories before samples are shuffled.</p>
				<h4>My suggested addition</h4>
				<p>Show one short pseudocode block matching <code>RolloutBuffer.compute_gae</code>, <code>get_minibatches</code> and <code>on_batch_update</code>. This will be more useful than a second generic PPO summary.</p>
			`,
		},
		{
			id: "386aa1bf-7b79-8026-8c08-fe9c063359b5",
			context: "Simulator Profiling",
			html: `
				<h4>Replace the TODO with the profiling conclusion</h4>
				<p>Across the recorded systems, rollout collection dominated PPO update time. On the main optimized 8-environment benchmark, collection took 32.1 seconds while the PPO update took about 1.0 second. This justified focusing on pathfinding and environment throughput before model-update optimization.</p>
				<h4>Include these benchmark groups separately</h4>
				<ul>
					<li>Single versus parallel environments on the same machine.</li>
					<li>Before versus after pathfinding optimization using the exact same command.</li>
					<li>CPU versus GPU PPO update on Kaggle, where the update fell from 65.1 seconds to 6.6 seconds at minibatch size 256.</li>
				</ul>
				<p>Do not compare raw frame times across machines without labeling hardware, OS, environment count, buffer size and minibatch size.</p>
				<h4>My suggested addition</h4>
				<p>Report steps per second as the primary number, then provide collection, GAE and PPO-update percentages. Readers can then see both system throughput and the remaining bottleneck.</p>
			`,
		},
		{
			id: "371aa1bf-7b79-8059-9ed3-c5bf6af32da5",
			context: "Experiment chronology",
			html: `
				<h4>Rewrite as phases, not a run dump</h4>
				<ol>
					<li><strong>Pipeline debugging, runs 1 to 6:</strong> simulator bugs, sparse reward, entropy collapse, too-small rollouts and diagnostic blind spots.</li>
					<li><strong>Fixed-opponent validation, runs 7 to 13:</strong> scripted bot, gamma comparison and the first step-penalty test.</li>
					<li><strong>Random-opponent scaling, runs 14 to 23:</strong> elixir masking, KL early stopping, normalization, rollout size and minibatch size.</li>
					<li><strong>Self-play and larger card set, runs 24 to 31:</strong> Elo bookkeeping fix, eight-card environment and architecture comparisons.</li>
				</ol>
				<h4>Correct causal language</h4>
				<ul>
					<li>Run 3 changed entropy weight, rollout size, parallelism and minibatch size. Do not attribute recovery to entropy alone.</li>
					<li>The run-9 claim that critic spikes came from rewards not being in 0 to 1 was a hypothesis, not a demonstrated cause.</li>
					<li>Run 24’s near-quadratic Elo growth was a checkpoint-rating bookkeeping bug, not rapid policy improvement.</li>
					<li>Run 25 versus 26 changed an architecture bundle. List the exact bundle instead of calling it simply “latest Deep Sets.”</li>
					<li>Run 30 versus 31 has suggestive curves and gameplay, but there is no shared frozen head-to-head evaluation. Do not rank the models conclusively yet.</li>
				</ul>
				<h4>Use the comments as ablation prompts</h4>
				<p>Fold the step-penalty, reward-weight and simulator-rule comments into the Ablations section. Keep this chronology focused on what changed, what was observed, what was retained and what remained uncertain.</p>
				<h4>My suggested addition</h4>
				<p>Annotate the existing curves with vertical callouts for elixir masking, gamma change, minibatch reduction, checkpoint Elo repair and architecture switches. Add a short failure montage beside the final gameplay clips.</p>
			`,
		},
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
