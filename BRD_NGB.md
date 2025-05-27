1.	INTRODUCTION
New Gen Bank in order to expand its cards business and also to explore business requirements in the schools sector has decided to launch strategic partnerships with schools to offer them a solution for school fee payments using NGB Credit cards via NGB channels. To Launch with, NGB has decided to extend services for school fee payment through NGB credit cards for Europe School.  

With this feature enabled, Customers who have NGB credit card can easily pay for the student’s school fee through Online banking, Mobile banking, IVR and E-Form [via Contact Center agent] and opt for converting the transaction to Easy Payment plan.


 
2.	SCOPE OF REQUIREMENT 

The Scope of the Requirement is as mentioned below:- 
A.	School Registration 
B.	Student Registration via Online Banking / Mobile Banking /E-Form [via Contact Center Agent] 
C.	Fee payment via Online Banking / Mobile Banking / IVR 
D.	Conversion of Fee Payment to EPP High level concept of process flow after school registration is shown below 
High level concept of process flow after school registration is shown below: 
  

A.	SCHOOL REGISTRATION:

•	An option to be made available for card operations team to register the school details. Following details to be available in the registration screen.
- School Name
- Location
- School Account Number with NGB
•	Product team should sent an approved email attaching necessary details for registration (mentioned above) in excel format to Operations team.
•	NGB GL account number to be configured internally in the system for GL settlement.
•	An option should be available to register the different type of Fee payments supported by each school. For eg: School Fee, Bus Fee, etc. These fee types should be listed in the fee payment page. 

B.	STUDENT REGISTRATION / AMENDMENT / DE-REGISTRATION:
•	An option to be made available to register the student details through Online banking, Mobile Banking and Contact center [using E-Form].
•	Customer should have an active credit card to proceed with the student registration.
•	More than one student can be registered for a customer and the student details provided should be captured under the customer RIM.
•	The student registered through Online Banking / Mobile Banking / Contact center should be saved in a common repository so that registered details are available across the channels.
•	Student details can be modified or de-registered from the registered student list.
•	SMS should be sent to customer during student registration / amendment / de-registration of a student. SMS should be in a predefined format with Student ID, Student Name and School Name. 

STUDENT REGISTRATION / AMENDMENT / DEREGISTRATION VIA ONLINE BANKING / MOBILE BANKING 
•	New option should be provided in online banking / mobile banking for Student Registration/ Student Deregistration
•	Following basic details are mandatory and have to be captured in student registration process 
- Student Name
- Student ID Number**
- School Name with Location (dropdown list with registered schools)
**Student ID Number has to be entered twice for verification purpose. Student ID field and  the Re-enter field should not allow copy paste option.
•	Student registration / maintenance should be authenticated via OTP validation
•	Student registration should happen only if the Student ID entered twice matches correctly. SMS should be sent to customer with the registration details – Student ID Number, Student Name and School Name.
Sample prototype for student registration via Online or Mobile Banking / E-FORM:-
 
STUDENT REGISTRATION / AMENDMENT / DEREGISTRATION VIA E-FORM [Contact Center Agent]
•	Customer calls the Contact center and authenticate himself with IVR TIN number
•	New option, School Fee Payment will be available under credit card section.
•	Once the “School Fee Payment” is selected, customer should be provided with two options – “Student Registration” and “Fee Payment”.
•	When “Student Registration” is selected from the options, Customer is redirected to Contact Center agent. Agent checks and verifies the RIM number of the customer.
•	Agent opens the new E-Form introduced as part of this requirement to enable Registration / Amendment / De-Registration of student. 
•	The RIM Number or Card number available in IVR has to be copied manually by the agent to the E-Form.
•	The E-Form should also list the previously registered students if available.
a)	If it is a first time user or a new student registration request, customer is requested to provide the student details for student registration process. Registration will be successful only if the student ID Number entered twice is matching. (These fields should not allow copy paste option)
b)	If it is a Amendment / De-Register Student request from customer, Contact Center agent should be able to select a student from the existing registered student list and modify or deregister on the E-Form. De-Register button should be on disabled mode by default and shall be enabled only on selecting a registered student from the list.
•	Customer can register more than one student on the same call. Option to register or modify should be available after each student registration or de-registration.
•	Customer should have an option to proceed for Fee Payment on the same call and Contact Center agent should refer those reuqests to “Fee Payment” option in IVR.
•	SMS should be sent in specific format to customer on successfully registering or modifying the student details with student ID, student name and school name.

C) FEE PAYMENT:
•	Payment can be done thru three options
- Via Online Banking
- Via Mobile Banking
- Via IVR
•	Customer can initiate fee payment only for the registered students. Each fee type against each student will be an individual entry.
•	Option should be there to convert the Fee payment to Installments [EPP].
•	Only active credit cards should be available for the payment and customer will have an option to select the credit card from the list (if multiple cards are available)
•	Customer can select the type of fee he wants to pay.
•	Payment to be allowed only if customer has sufficient balance.
•	System should maintain previous transaction logs and transaction details with unique 
•	reference number and description.
•	Confirmation SMS should be sent out to customer on successful payment and EPP conversion with the payment details [Student Name, School name, Fee Type and the Amount].
FEE PAYMENT VIA ONLINE BANKING / MOBILE BANKING
•	New option should be provided in online / mobile banking portal for school fee payment.
•	Customer logs in to online banking / mobile banking and selects the new option provided for School Fee Payment.
•	Customer can choose a credit card from the dropdown list. Available balance should be shown when a credit card is selected.
•	Registered students under the customer RIM and different types of fees registered for the selected school should be displayed in a dropdown selection.
•	Customer can select the student, fee type from the dropdown and add the amount, remarks(Max 20 characters). If sufficient balance is available to complete the transaction, this item to be added to the payment list.
•	From the payment list, customer should have an option to convert the payment to EPP.
•	Customer can select individual payment from the list for EPP conversion with the plan details. 
•	An E-form Service request with the Transaction information should be created (using unique ref no & card id) and assigned to Card Centre separately for each EPP selected transactions in the list.
•	Payment should be initiated only if sufficient balance is available in the selected credit card. 
•	Payment for more than one student can be initiated or processed together.
•	Each Payment type for each student to be an individual entry in cards and school account.
•	Confirmation SMS should be sent out to customer on successful School fee Payment. SMS should be in specific format and should contain the transaction details (School Name, Student Name, amount and the Fee Type)
•	Last 6 month history of fee payment should be available in the system. 
Sample prototype for Fee payment & EPP Conversion via Online or Mobile Banking / E-FORM:-

 

FEE PAYMENT VIA IVR

•	Customer calls the Contact center and authenticate himself with IVR TIN number.
•	New option, School Fee Payment will be available under credit card section.
•	Once the option is selected customer is provided with two options – “Student Registration” and “Fee Payment”.
•	Customer to select “Fee Payment” option to pay the school fee of already registered students.
•	Once fee payment option is selected, customer will be provided with the options to select school, student and school fee type.
•	Customer to follow the below steps after selecting school fee payment option for making the payment on IVR:
a)	Select Student- IVR to voice out the student ID and customer to make selection for identifying the student for payment. (The student name can be voiced out but will require to implement a new IVR module(text to speech) which requires to be checked for cost and feasibility from IT/vendor)
b)	Select Type of Fee- IVR to provide menu options to customer for choosing the type of fee that customer wants to pay e.g., tuition fee, bus fee, other fee 
c)	Enter Amount-Customer to enter the amount to be paid 
d)	Payment done and SMS sent to the customer
e)	Provide Menu option to the customer to make another payment for the same student or another student. Once all payments are done customer to be provided an option to convert the transactions into EPP. If customer opts for EPP call will be transferred to an agent for initating the EPP request.
f)	ICRS to be updated with students registered and payments done by the customer with details
g)	Daily Payment limit on IVR and OTP options for payment to be configurable by business,if required.
•	Payment will be initiated by the IVR with the details provided by customer.
•	Contact Center agent will create an E-Form (for each transaction), siting the authorization in Vision Plus (Please note that description of the transaction will not be available in Vision Plus. Transaction description will be like “MEMO POSTED CREDIT”).
•	The E-Form created will be queued in Contact Center queue until the transaction is posted (max of 2 days).
•	Once the transaction is posted, the E-Form created by agent will be verified at Contact Center itself before it is sent to cards center for processing.
•	If EPP option is not selected, call is ended with the confirmation of the payment details.
•	Confirmation SMS should be sent out to customer on successful School fee Payment in specific format with payment details (school name, student name and school fee type).
Note: Contact Center will be responsible for the verification of EPP transaction details forwarded to Cards center thru E-Form.
FEE POSTING
•	When customer initiates a payment, amount should be debited individually from the customer’s credit card and credited individually to the registered school’s account. GL posting also should happen in real time. Each transaction should have a proper description. The posting steps include:
-	Debit Customer Credit card for individual fee with description as < SCHOOL NAME > -< FEE TYPE > - < STUDENT ID NUMBER>
-	Debit GL account [Separate for Visa Conventional, Visa Islamic & MasterCard] (configured for school fee payment) for individual fee in Corebanking System with description as “School Pymt-cardid-<UNIQUE REF NUMBER>”
-	Credit to School Account for individual fee payments in Corebanking System with description as “< STUDENT ID NUMBER>- < FEE TYPE > -<UNIQUE REF NUMBER>”
•	Maximum length allowed for transaction description is 40 characters. Some of the parameters in description should be trimmed off to meet the maximum length constraint. 
Note:
•	New transaction code shall be used for the transaction postings. AUTH code will be used as Unique Reference Number across the systems.
•	School website can have a ‘School Fee Payment’ link on their website which should open a the login page in Online banking with an option for school fee payment. 
•	In case of different multiple payment scenario, Credit card should be debited individually for each student and fee and the school account also has to be credited individually. That is, Individual posting entry should happen for each student and fee type even if there are more than one payment.
For eg:
First posting: Dr Credit Card of Customer for student 1 – fee type
-	Dr GL Account for student 1 – fee type
-	Cr School account for student 1 – fee type
Second posting: Dr Credit Card of Customer for student 2 – fee type
-	Dr GL Account for student 2 – fee type
-	Cr School account for student 2 – fee type
•	On the next day of payment, an excel file should be generated and sent via email to the school with below details. The mail should be triggered automatically to the configured distribution list. The file should be generated separately if multiple schools are available.
-	School Name with location
-	Student ID
-	Student Name
-	Amount
-	Transaction date
-	Payment Remarks [value entered by the customer in the Remarks field in Fee 	Payment page]
Sample format is given below:
 
•	While initiating a payment, total fee payment amount entered is validated against the available balance [OTB] of the credit card selected. Payment cannot be initiated if the balance is not available.
•	All the transactions should be logged and transaction details with unique reference number and description has to be saved

ASSUMPTIONS
•	Schools register with NGB for school fee payment will have NGB account number.
•	Loyalty points will not be granted for the School Fee payment if EPP option is availed.
•	If sufficient balance is not available during EPP conversion (due to other payment settlement or auto debit), the EPP request should be rejected and customer should be notified via SMS in a specific format with sufficient details.

OUT OF SCOPE
•	Any integration to school’s website for Fee information retrieval is out of scope.
•	Any Integration to any other payment gateway is out of scope.

BUSINESS UNITS INVOLVED
•	CARDS Management
•	Direct Banking
•	Contact Centre

SYSTEMS INVOLVED
•	Online Banking
•	Mobile Banking
•	CRM
•	IVR
•	Cards System
