# Research plan: composing independently trained SO-100 policies

This is a proposed experimental design, not a claim that the method works or is novel. It assumes simulation access now and two SO-100 arms later. The main objective is to measure how much additional learning is needed to coordinate reusable single-arm skills, and where that reuse fails.

## 1. Ask a narrow, falsifiable question

**Can a small coordinator control two frozen, independently trained, goal-conditioned arm policies on cooperative manipulation tasks using less additional interaction than joint fine-tuning, while retaining performance when an arm policy is replaced?**

Define these terms before implementation:

- **Independent:** each arm policy is trained without the other arm, joint trajectories, or a cooperative reward. A shared architecture, interface, and training distribution are allowed and disclosed.
- **Frozen:** no updates to policy weights, observation normalization, or other learned preprocessing during coordination training or evaluation. Save checkpoint hashes and freeze running statistics.
- **Composition:** the coordinator supplies goals, timing, and explicitly declared mode commands; the individual policies continue generating arm actions.
- **Replacement:** deploy the same coordinator with another independently trained compatible policy, without adapting either component.
- **Efficient:** specify interaction budget, wall-clock training time, and achieved success. Report base-policy acquisition costs separately and in total.

This cannot establish whether arbitrary pairs of black-box policies can always cooperate. The policies need suitable capabilities and an interface through which behavior can be directed. A negative experiment only rules out the tested method, capabilities, tasks, and budgets.

## 2. Position the contribution against prior work

The broad idea already has precedents:

- [Synchronize Dual Hands for Physics-Based Dexterous Guitar Playing](https://arxiv.org/abs/2409.16629) trains hand policies separately and coordinates them through latent-space manipulation. Its method freezes pretrained policies and optimizes a synchronizer. This is a particularly close conceptual precedent, though it studies simulated dexterous hands playing guitar.
- [AnyBimanual, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_AnyBimanual_Transferring_Unimanual_Policy_for_General_Bimanual_Manipulation_ICCV_2025_paper.html) transfers pretrained unimanual policies using a skill manager and visual aligner, with bimanual demonstrations.
- [TwinVLA](https://arxiv.org/abs/2511.05275) composes copies of a pretrained single-arm VLA and studies data-efficient bimanual adaptation. Copying one model is a different experimental setup from independently training two interchangeable controllers.
- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201) motivates combining structured control with a learned correction; it is not evidence that this particular dual-arm design will work.

A plausible contribution is a small, reproducible study of **coordination cost, policy interchangeability, and failure boundaries on inexpensive robot arms**. Check the full papers and their references before claiming novelty. Changing hardware or substituting an MLP is not sufficient novelty by itself.

## 3. Reduce the number of simultaneous research problems

Start with state-based simulation: joint measurements, end-effector poses, object pose, and task phase. Use MuJoCo with a verified SO-100 model if available. Validate joint axes, limits, actuator behavior, gripper contact, coordinate transforms, and shared reachable workspace before training.

Use real physics for cooperative interaction. Do not weld an object to a gripper or impose artificial constraints that give the controller cooperation for free. Any simplified contact assumptions must be disclosed and tested separately.

Avoid learning perception initially. On hardware, the existing RealSense/ArUco tools can provide object pose after camera-to-base calibration and uncertainty checks. A simulation method that consumes perfect state must later be tested with measured noise, delay, and missing detections.

Limit task orientations to configurations the SO-100 can actually reach. Treat arbitrary orientation control, tight insertion, deformable objects, and rich force control as later projects. Do not make completing sim-to-real transfer a prerequisite for asking the coordination question in simulation.

## 4. Design the policy interface before training

Use a common interface such as:

    action_i = policy_i(local_observation_i, goal_i, mode_i)

A goal specifies an achievable local target; modes such as approach, grasp, hold, and release exist only if the policy was explicitly trained to support them. Train over varied initial conditions, target positions, modest object variation, and moving or paused goals. Policies trained for one fixed trajectory may be impossible to redirect meaningfully after freezing.

Choose a small skill set that covers the proposed cooperative task. Independently trained pick-and-place skills do not automatically include receiving an object from another gripper, carrying shared loads, or maintaining grasp under another arm's motion. Probe those capability boundaries explicitly.

PPO is acceptable if you already have a working implementation. It is not part of the scientific hypothesis. If base-policy creation becomes the bottleneck, compare learning from scripted single-arm demonstrations with RL; report how those demonstrations were generated and their cost. Keep the acquisition method fixed during the main coordination comparison.

First validate one policy, then train independent instances for the two arms. Use a declared held-out single-arm test set and a prespecified competence threshold. A practical starting target is around 90% success, but choose it for the task rather than treating it as a universal standard. Retain all training-run outcomes and disclose any selection.

## 5. Establish cooperation with no new learning

Build a feedback-based state machine: approach, grasp, confirm readiness, move, place, release. Advancement depends on observed readiness, not only elapsed time.

For cooperative transport, define both grasp targets relative to the same desired object pose. Transform those targets into each arm's base frame. Slow or pause shared progress when one arm falls behind. Calibrate the transforms; a shared world coordinate system is not learned coordination.

Implement this controller both over frozen learned policies and over a conventional IK/position-control backend. The latter tests whether the task needs learned single-arm policies at all.

If the structured controller already reaches the target success rate, report that result. Add disturbances and held-out conditions to determine its operating limits. Do not deliberately weaken it to make a learned component look useful.

## 6. Add only the smallest useful learned coordinator

Before training a network, try a small declared search over coordinator parameters such as approach offset, shared speed, and readiness tolerance. Count every development rollout in its tuning budget and lock the parameters before final evaluation. If this suffices, you have found a cheaper solution; compare it fairly with learned alternatives.

If state-dependent corrections are still needed, start with a small MLP, for example two hidden layers of 64 units. Inputs can include both goal-tracking errors, object pose error, current task phase, and previous coordinator output. Outputs should initially be bounded goal corrections and a shared progress rate. Keep gripper sequencing in the same state machine across compared methods.

Conceptually:

    nominal_goals = geometry_and_task_phase(observation)
    corrections, progress_rate = coordinator(observation)
    arm_goals = feasible_bounded_goals(nominal_goals + corrections)
    arm_actions = frozen_arm_policies(local_observations, arm_goals)

Select spatial bounds from the verified workspace and supported training-goal distribution. Run the coordinator more slowly than the low-level control loop, using the same rates for all relevant baselines. Record simulated time and low-level interactions, since counting only coordinator decisions would artificially improve sample-efficiency numbers.

First determine which failures the coordinator is meant to fix: timing mismatch, grasp-position bias, or object drift. A small supervised imitation model can be used if successful coordination targets are available; otherwise train the small coordinator in simulation using the existing RL implementation. Neither choice guarantees fewer interactions. Count all demonstrations and training data.

Keep the initial correction in goal space. Direct motor residuals can bypass the frozen skills and change the interpretation of the experiment. If you later add action residuals, label them as a separate method and constrain them explicitly.

Do not add graph attention until the MLP's failures establish a need. With only two fixed arms, richer architecture is an additional hypothesis, not a prerequisite.

## 7. Use a manageable task ladder

| Task | Purpose | Success must establish |
| --- | --- | --- |
| Single-arm pick-and-place | Qualify the reusable modules | Reliable completion over held-out starts/goals |
| Object handover | Test sequential cooperation and release timing | Receiver retains the object and finishes placement without a drop |
| Shared-object lift and transport | Test simultaneous physical coupling | Both arms contribute; object stays within tilt/drop limits and reaches its destination |

Start with one cooperative task. A handover needs a receiver skill that works in the handover region; test this before blaming the coordinator. For transport, use a lightweight object with two accessible handles and a reachable path. Demonstrate that the chosen setup actually requires cooperation, using single-arm or disabled-partner controls. Do not use an excessive load to force the requirement.

A robust study of one task family, multiple variations, and policy swaps is preferable to several unreliable demonstrations. Add the second cooperative family once the first is reproducible; keep the paper's scope aligned with the actual task coverage.

## 8. Compare methods that answer distinct questions

| Method | Additional learning | Question |
| --- | --- | --- |
| Geometry/state machine + IK | None | Does conventional control solve the task? |
| Geometry/state machine + frozen policies | None | Are reusable skills already sufficient? |
| Same scaffold + small coordinator + frozen policies | Coordinator only | Does learned coordination improve robustness efficiently? |
| Same pretrained policies and scaffold, with joint fine-tuning | Coordinator and arm policies | What performance is lost by freezing? |
| Joint policy trained from scratch | All policy parameters | Does reuse reduce total acquisition cost? |

Uncoordinated execution is a useful diagnostic, but is too weak to be the sole baseline. Keep sensing, available task information, reward, limits, evaluation conditions, and tuning budgets comparable. Distinguish system-level comparisons from controlled ablations where only one component changes.

Implement the zero-learning methods first, then the frozen coordinator and matched fine-tuning comparison. Add the from-scratch reference for claims about pretraining savings. When a baseline does not converge within budget, report that bounded result; do not call it incapable of solving the task.

## 9. Test interchangeability directly

Train several single-arm instances independently, for example L1–L3 and R1–R3. Train a coordinator on L1/R1 and evaluate it unchanged on the other pairings. The policy observation/action interface must remain identical. Changing robot geometry or policy architecture is a separate, harder generalization claim.

Report both absolute success and loss relative to the original pair. Qualify each replacement's single-arm competence so that a broken replacement is not mistaken for a coordination failure. State the limited number of independent modules; nine pairings built from six modules are not nine independent training experiments.

This swap test distinguishes learning one pair's habits from learning a reusable coordination strategy. If it fails, report pair-specific coordination and investigate goal conventions, normalization, speed, and grasp behavior before adding a larger network.

## 10. Evaluate rigorously within a small budget

Create separate development and final evaluation sets. Fix final object placements, disturbance ranges, success criteria, deadlines, and training budgets before final runs. Tune using development results only, select checkpoints using a declared rule, and use the same evaluation starts across methods.

For development, use a small number of seeds. For the final learned-method comparison, budget roughly five independent training runs per method/task as a starting point, increasing this if pilot variation is large and resources allow. This is not a guarantee of statistical power. Repeated episodes from one checkpoint cannot substitute for independent training runs.

A practical simulation target is 100 held-out episodes per trained run for the main condition. Run selected perturbation tests on the strongest relevant methods rather than multiplying every possible experiment. Report per-seed outcomes and uncertainty across training runs; do not pool all episodes into a falsely precise estimate of training reliability. Confidence intervals with very few seeds remain uncertain. See [Agarwal et al.](https://arxiv.org/abs/2108.13264) and [Colas et al.](https://arxiv.org/abs/1806.08295).

Recommended robustness conditions:

- Nominal held-out placements and goals.
- Unseen combinations of object mass/friction within declared bounds.
- Asymmetric arm delay or speed reduction.
- Object-pose noise and short observation dropouts.
- Replacement arm-policy pairs.

Test individual disturbances first; use a small prespecified combined stress set later. Report failure categories: perception, reachability, grasp, timing, collision, and shared-contact instability. Avoid silently excluding failed trials.

For hardware, start with 20–30 trials per selected method/task if practical, report exact successes/trials with binomial uncertainty, and randomize method order. These are a limited transfer check, not strong evidence of rare-failure safety or broad reliability. Record interventions and failed resets separately; a human rescue does not count as autonomous success.

## 11. Measure efficiency without hiding costs

The main figure should show task success versus additional cooperative environment interaction, with uncertainty across training runs. Also show wall-clock training time on the stated hardware. Choose a success target before final runs, for example 80%, and report the first scheduled evaluation/checkpoint meeting the declared criterion. Methods that never reach it are marked as not reached within budget.

Maintain two cost accounts:

1. **Adaptation cost:** coordination learning given already available arm skills.
2. **Total cost:** skill acquisition, coordination learning, demonstrations, and relevant tuning costs.

Record simulated seconds, low-level control steps, training hardware, parallelism, resets, and human data-collection effort. Parameter count is useful context, but does not establish faster learning. If pretraining only becomes economical after reuse across several tasks, show the cumulative cost across those tasks; do not assume that break-even occurs.

## 12. Use decision gates rather than an ambitious calendar

1. Validate simulation and conventional controllers; establish feasible contacts and workspace.
2. Validate single-arm skills and interfaces; freeze qualified checkpoints.
3. Run the structured frozen-policy baseline; catalogue the failures.
4. Train the small coordinator and matched fine-tuning baseline under fixed budgets.
5. Run final seeds, held-out perturbations, and policy swaps.
6. Transfer the strongest simple methods to hardware when available.
7. Write conclusions bounded by the actual results.

If base policies lack a necessary skill, return to the capability definition and disclose the additional training. If geometry solves all tested conditions, the data do not support a need for learned coordination. If fine-tuning substantially outperforms frozen composition, investigate the lost capability rather than hiding the result. If no approach works, check task feasibility and calibration before attributing failure to modularity.

## 13. Write the paper around evidence

A provisional title is **How Much Coordination Learning Do Frozen Single-Arm Policies Need?**

Write the problem definition and experimental protocol before final results. Then organize the paper around: related work and precise gap; policy interface and task assumptions; structured baseline and small coordinator; training-cost curves; policy-swap and robustness results; hardware transfer if available; and limitations.

Useful figures are the system diagram, learning curves with uncertainty, a matrix of swap results, and a compact failure breakdown. Release environment/config versions, seeds, checkpoints, evaluation starts, raw outcomes, and plotting code where practical.

A defensible conclusion might say that a particular interface and coordinator achieved a measured success level using a measured adaptation budget, under specified task and replacement conditions. It should also identify which conditions required fine-tuning or new single-arm skills. The numbers must come from experiments; neither positive results nor publication are guaranteed.
