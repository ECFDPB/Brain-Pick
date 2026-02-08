# Brain-Pick

ETH Oxford 2026 Project - Utilising EEG and eye movement sensors to find the interest point within rich media contents.

## Background

**Are you also troubled by "probably liking"?**

Today’s rich media platforms mostly infer user interests from clicks, taps, and viewing history—approximations that only reflect what users do, 
not what they truly think or feel. These indirect signals often lead to inaccurate recommendations, repetitive content, 
and persistent mismatches between platform suggestions and real user preferences.

We aim to change this by extending traditional behavioral analytics to cognitive and physiological response analysis: 
understanding not just actions, but genuine attention, emotion, and implicit preference.

The user would wear smart glasses with EEG sensors and 2 cameras integrated.
The rear camera will track the user's eye movement to analyse where the user is looking, 
and the EEG sensors will be used to monitor users' brainwaves and determine the level of like or dislike for the contents.

By leveraging this technology, media platforms can better identify users’ genuine needs and deliver content that is more closely aligned with their underlying physiological preferences. 
As a result, the latency of users’ behavioral feedback can be reduced from 3 seconds to just 0.5 seconds, significantly enhancing user satisfaction and increasing platform retention rates.

[![Demo Video](https://res.cloudinary.com/dvlc7v3l8/image/upload/v1770545516/demo_graph_cgdhnz.png)](https://res.cloudinary.com/dvlc7v3l8/video/upload/v1770545378/Demo_deb0ec.mp4)
<small>The Demo video of our system. Click the image to view it.</small>

## Business Logic

In the initial phase, mainstream media platforms can sell low-cost smart glasses at a low price to their users, 
and encourage user participation in the program by signing cooperation agreements or offering exclusive discounts and perks. 

After accumulating sufficient user physiological data, 
businesses can call the system's server API to accurately obtain users' genuine physiological demands and preferences, 
improving the decision-making efficiency of content operation and commercial delivery by 60% while reducing trial-and-error costs. 

In the future, with users' explicit authorization, media platforms can provide desensitised physiological preference datasets to partners,
offering scientific data support for targeted advertising, content planning, product R&D and iteration, and consumption trend analysis, 
thus achieving compliant data monetization.

![EEG Pipeline](https://res.cloudinary.com/dvlc7v3l8/image/upload/v1770546271/Pipeline_paper_i8catr.png)
<small>The pipeline of our EEG analysis system.</small>
![Product Sketch](https://res.cloudinary.com/dvlc7v3l8/image/upload/v1770544458/glasses_y5fbnv.png)
<small>The sketch of our simply smart glasses.</small>

## Privacy Protection

With privacy issues in mind, we promise that clients are free to decide whether to enroll in such a model or not.

During the programme, all raw physiological and behavioral data is processed locally on the user’s device, and only encrypted, 
anonymous interest tags are uploaded to the server via TLS/SSL.

In addition, datasets shared with third parties are fully de‑identified, containing no personally identifiable information. 
Throughout the entire process, users retain full access to and control over their data, and they can delete any of their data at any time if they wish.

![Multimodal Data Processing Flow](https://res.cloudinary.com/dvlc7v3l8/image/upload/v1770544978/encrypted_ewexic.png)
<small>The multimodal data processing flow of our system.</small>

## Demo

The demo does the following things:

- The clients fetch pages from the server and display them to the user.
- The users' EEG and eye movement data are collected via sensors on the client device and analysed locally. The eye movement data provides insights into what the user is looking at and what the content is about, while the EEG data is analysed to predict how the user feels about that content.
- The clients aggregate this information into a structured, encrypted report and send periodic updates to the server.
- The server persists all user reports in a database, and clients can access and view their own data reports at any time.
- Using these insights, existing rich media platforms can offer more personalised services to users.
