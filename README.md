###Welcome to LinkedGuard! 
##What's LinkedGuard all about?
We’re tackling a big problem on LinkedIn—fake companies that post job ads to steal personal info or commit identity theft. And, there’s also the issue of these phony profiles snagging data from resumes. Not cool, right? So, we built LinkedGuard to keep your LinkedIn experience safe and secure.
##How does LinkedGuard work?
LinkedGuard is split into two cool parts:
#1.	Checking if Companies are Legit: We use a Machine Learning (ML) model to figure out if a company on LinkedIn is real or just a fake. We base our predictions on a bunch of data points and use something called Decision Trees to see if we should trust a company or not. To make sure our model isn’t biased towards the usual suspects, we use a technique called SMOTE to even things out. This helps our model be fair and accurate. We double-check our work with k-fold cross-validation to make sure we’re on point across different sets of data.
#2.	Smart Resume Scraper: Based on what our ML model says about a company's risk, we’ve got a scraper that smartly picks data from resumes. We set up clear rules on what to scrape and why, focusing only on what’s necessary based on the company's risk level.
##How did we figure this all out?
We did a bunch of homework! We sent out surveys in classes and neighborhoods to see what features make a company seem legit. After gathering responses from a diverse group of 45 LinkedIn users, we analyzed the data both ways—numbers and words—to pick out the 11 best features that help us predict if a company is the real deal.
##Conclusion
LinkedGuard is here to make your LinkedIn experience safer. By smartly identifying risky companies and managing how resumes are shared, we’re putting privacy and security at the forefront of your professional networking. Dive in and see how we’re making LinkedIn a safer place for everyone’s career growth!

Read our full report on :  https://github.com/saravan-p/LinkedIn_Safeguard/blob/main/Linked_Guard.pdf
