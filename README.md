Lets figure ts out
Here is the deal, this project is Messing around with the So100 for ARC and Abhishek Sarkar, but from the inside it has these timelines:
	By End of this Semester: Both the physical Arms are fixed, implement Normal Single arm manipulation on both those arms, and have a solid implementable plan for dual arm manipulation(which shall be discussed in this doc as well later… )
	By Midsem of Sem2: Implement and make better said plan for dual arm manipulation( obviously some problems will be encountered which may demand a complete pivot also soo)
	By end of Sem 2: Derive “research related insights” maybe actually come up with something impactful… 

Now for the plan: 
Since we need to train both the arms separately with PPO by the end of this sem, its preferred arm dual arm manipulation strategy such that if i have two pretrained RL arms, they can modularly connect with each other and work independently and together. Is this possible is the next question we need to think about…

At first it found out about something called Modular Deep Learning:
	https://arxiv.org/abs/2108.00385 : this one was imitation learning, since in my goal, i had mentioned that we’re trying for a method, where if two robotic arms, who have never seen each other need to work together to pick something up, (given that they are both trained to pick things up in that environment). In imitation learning like this, i am pretty sure i will need train them by picking stuff up which they'll imitate together.. In this paper they make a self-attention mechanism which seems like it’ll help the rl converge and will be very helpful in the real plan..(which shall be decided in this doc later .. )

	
	https://drive.google.com/file/d/1OVHgc8fWEyFbpnN4hcHjDKf2YabkL2xT/view?usp=sharing
: this one is something called “Deep multimodal Learning Framework”, i saw multimodal in the title, thought maybe something relevant maybe.. these guys arent even going for dual arm manipulation (single arm) but why i have put this in is because they use image based deep RL and seem to have great results (I wouldnt really know though, Arjun, is there anything that could be really useful in the way they use pretrained vision encoder, maybe we could use one too..)
One really important reason why this paper is included in the doc is because they mention freezing pretrained models, which we are thinking about in the real plan (which shall be decided in this doc later .. )

These two papers are directly related to what we’re doing and we’re basically going to be doing what they did for that specific part of the project..  So its realy important for me to read these, but i havent done that yet.. I will do it after atmos.:

https://proceedings.mlr.press/v229/jian23a.html
https://arxiv.org/abs/2003.09540

One policy to control them all paper:
https://proceedings.mlr.press/v119/huang20d.html




Da real Plan : 

Train two SO-100 policies independently in MuJoCo using PPO, just like in the sims
Sim2Real Transfer both policies and establish reliable single-arm manipulation.
Freeze both policies and construct a dual-arm cooperative manipulation environment.
Establish a geometry-based coordinates baseline using object-relative grasp targets.
make a lightweight learned coordinator, initially an MLP and evolve it into a graph-attention architecture, while keeping the base policies frozen.
Hopefully this works





