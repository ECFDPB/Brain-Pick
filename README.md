# Brain-Pick

ETH Oxford 2026 Project - Utilising EEG and eye tracking to find the interest point within rich media contents.

## Background

We want to provide an extension to existing rich media platforms.
Currently, most of these platforms try to catch the users' interests by analysing what they clicked on.
We want to extend this to analysing what the user's actually thinking and responding to the content.

The user would wear smart glasses with EEG sensors and 2 cameras integrated.
The rear camera will track the user's eye motions to analyse where the user is looking at.
The front camera will be used to track the content on the user's devices.
    However, because we lack working smart glasses,
    we'll assume that the client is running fullscreen on the user's device,
    hence the absolute coordinates we got is just the relative coordinates in the page.

## Business Logic

With privacy issues in mind, we think this should be an optional plan to users,
they have the choice whether to enroll in such a model.

The platform can offer free or reduced membership plans as an exchange to users providing their data.

With the user's consent, the platform can also sell the data to third parties.

## Demo

The demo does the following things:

- The client gets a page from the server and displays it to the user.
- We get the user's EEG and eye tracking data via sensors on the client device.
- The eye tracking data provides where the user is looking at, then the client gets the most likely element and find its tags.
- The client analyses the EEG data to predict how the user is feeling about these tags.
- The client combines these as a report to the server, the reports are taken regularly.

- The server stores all reports in a database.
- The user can get their data reports anytime.
- Based on these information, the existing rich media platform can provide better services to the user.
