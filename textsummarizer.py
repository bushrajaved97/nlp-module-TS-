# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:48:48 2019

@author: bushr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:01:16 2019

@author: bushr
"""

import re
import nltk
import heapq
from twilio.rest import Client
import smtplib
import requests
from requests.exceptions import HTTPError
import speech_recognition as sr

text="I have been suffering from hormone issues since months. You may be suffering from invasive cancer. I will gave a regimen for radiation therapy. This is a standard treatment to detect tumor markers. We will give you calpol and panadol. Thank You Doctor."

#preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)

sentences = nltk.sent_tokenize(text)

stop_words = nltk.corpus.stopwords.words('english')    

#Figuring out cancer terms and medicine prescribed in the conversation
#tokenizing the sentences
words = nltk.word_tokenize(text)

#applying part of speech on each individual word
tagged_words = nltk.pos_tag(words)

#collecting all the nouns in a seperate list
word_tags=[]
for tw in tagged_words:
    if tw[1] == "NN" or tw[1] == "NNS" or tw[1] == "NNP" or tw[1] == "NNPS":
        word_tags.append(tw[0])
        
        
#fetching cancer terms meaning from the API's
health_term={
"antibodies":	"Proteins in the immune system that recognize and attach to foreign molecules, called antigens",
"benign":	"A tumor that is not cancerous. The tumor does not invade nearby tissue or spread to other parts of the body",
"biopsy":	"Removal of a tissue sample that is then examined under a microscope to check for cancer cells",
"carcinoma	":"Cancer that starts in skin or tissues that line the inside or cover the outside of internal organs",
"chemotherapy":	"Drugs used to destroy cancer cells by interfering with their growth and/or preventing their reproduction",
"clinical trial":"Research studies that test new treatment and prvention methods to find out if they are safe, effective, and better than the current standard of care (the best known treatment)",
"phase I clinical trials":	"Tests new types of treatment and aim to define a safe dose that will be used for further studies. This is usually the first testing of a treatment on humans after extensive laboratory work. Recruitment for Phase I trials are usually from patients for who...",
"phase II clinical trials":"Test the anti-cancer effects of the new treatment, and include very detailed toxicity investigations. If there is effective antitumour activity, it may be incorporated in a future phase III study",
"phase III clinical trials":	"Compare one or more treatments of proven efficacy. Often patients will be randomized between an established 'standard' treatment and a new 'experimental' treatment - it is not known which the better treatment is",
"endpoint":	"The results measured at the end of a study to see if a given treatment has worked",
"epidemiology":	"The study of the patterns, causes, and control of disease in groups of people (The study of cancer may involve looking at how many people have cancer; who develops specific types of cancer; and what factors, such as genetics or personal behavior, play a ...",
"gene":	"A length of DNA that carries the genetic information necessary for production of a protein. Genes are located on chromosomes and are the basic units of heredity",
"genetic testing":	"The analysis of a person’s DNA to check for genetic mutations (changes) that carry an increased risk of or predisposition to cancer",
"hormone":"A substance produced by an organ or gland that is carried by the blood and produces a specific effect on other organs or glands",
"malignant":	"A tumor that is cancerous. The tumor may invade nearby healthy tissue or spread to other parts of the body",
"metastasis":	"The spread of cancer from where the cancer began to another part of the body(Cancer cells can break away from the primary tumor and travel through the blood or the lymphatic system to the lymph nodes, brain, lungs, bones, liver, or other organs)",
"oncogene":	"A normal gene that when mutated plays a significant role in causing cancer",
"palliative":"Treatment of the physical, spiritual, psychological, and social needs of a person with cancer. Its purpose is to improve quality of life ",
"mutations":	"changes",
"precancerous":	"Changes in cells that may, but do not always, become cancer. Also called pre-malignant",
"predisposition":"A tendency to develop a disease that can be triggered under certain conditions (Predisposition to cancer increases a person's risk of developing cancer, it is not certain that the person will develop it)",
"acute":	"Symptoms that start and worsen quickly but do not last over a long period of time",
"cbc":	"A test to check the number of red blood cells, white blood cells, and platelets in a sample of blood",
"platelets":	"Components of blood that help it to clot",
"chronic":"A disease or condition that persists or progresses over a long period of time",
"insitu":	"Cancer that has not spread to nearby tissue. Also called non-invasive cancer",
"invasive cancer":"Cancer that has spread outside the layer of tissue in which it started and is growing in other tissues or parts of the body. Also called infiltrating cancer. Localized cancer: Cancer that is confined to the area where it started and has not spread to oth...",
"neutropenia":	"An abnormal decrease in the number of neutrophils in the blood. Neutrophils are a type of white blood cell that fights infection",
"prognosis":	"Chance of recovery; a prediction of the outcome of a disease",
"protocol":	"An action plan for how a clinical trial will be carried out. It states the goals and timeline of the study, who is eligible to participate, what treatments and tests will be given and how often, and what information will be gathered",
"regimen":	"A treatment plan that includes which treatments and procedures will be done, medications and their doses, the schedule of treatments, and how long the treatment will last",
"stage":	"A measurement given or a diagnosis that describes the size of the original tumor and identifies whether the tumor has spread to lymph nodes or other parts of the body",
"standard of care":	"A set of common guidelines that is followed for the diagnosis and treatment of a certain type of disease",
"adjuvant therapy":	"Treatment given after the main treatment. It usually refers to chemotherapy, radiation therapy, hormone therapy, or immunotherapy given after surgery to reduce the chance of cancer coming back",
"biologic therapy":	"Treatment to stimulate or restore the ability of the immune system (the body's defense) to stop or slow the growth of cancer cells or help control side effects. (Also called biologic therapy, immunotherapy, or biologic response modifier [BRM] therapy) ",
"chemoprevention":	"The use of drugs, vitamins, or other agents to reduce the chance of developing cancer or having cancer come back",
"complementary and alternative medicine":"CAM is a term used to describe a diverse group of treatments, techniques, and products that are not considered to be conventional or standard medicine",
"complementary medicine": 	"It is used in addition to conventional treatments (an approach that is also called integrative medicine",
"alternative therapies"	:"They are unproven treatments used instead of standard treatments",
"hormone therapy":	"Treatment that removes, blocks, or adds hormones to kill or slow the growth of cancer cells. (Also called hormonal therapy or endocrine therapy)",
"neoadjuvant therapy":	"Treatment given before the main treatment. It may include chemotherapy, radiation therapy, or hormone therapy given prior to surgery to shrink a tumor so it is easier to remove",
"radiation":	"The use of high-energy rays (such as x-rays) to kill or shrink cancer cells",
"external radiation":	"The radiation come from a machine outside the body ",
"brachytherapy or internal radiation":	"The radiation comes from a radioactive materials placed in the body near cancer cell",
"standard treatment":	"Treatment that has been scientifically tested, found to be safe and effective, and approved by the U.S. Food and Drug Administration (FDA). (Often called conventional treatment)", 
"targeted treatment":	"A form of cancer therapy that takes advantage of the biologic differences between cancer cells and healthy cells by “targeting” faulty genes or proteins that contribute to cancer growth. The treatment blocks the spread of cancer cells without damagin...",
"follow-up care plan":	"A patient's plan written oncologist following treatment that summarizes the therapy(ies) and outlines long-term care needs.(This may include how often the person needs to see a doctor and any future tests needed. It may also include advice on healthy lif...",
"late effects":	"Side effects of cancer treatment that appear months or years after treatment has ended. (This may include physical and mental problems, as well as development of secondary cancer)",
"progression-free survival":	"The length of time during and after treatment that the cancer does not progress or grow",
"recurrence":	"Cancer that has returned after a period of time when the cancer could not be detected",
"local recurrence": 	"It means that the cancer has come back to the same place as the original cancer",
"regional recurrence": 	"It refers to cancer that has come back after treatment in the lymph nodes near the original cancer site",
"distant recurrence": 	"It is when cancer spreads after treatment to other parts of the body",
"relative survival":"The length of time after treatment that a person with cancer lives, excluding all other causes of death but cancer",
"remission":	"The disappearance of the signs and symptoms of cancer but not necessarily the entire disease. The disappearance can be temporary or permanent",
"complete remission": 	"It means all known tumors have disappeared",
"partial remission": 	"It refers to a greater than 50% reduction of tumor mass",
"morbidity":	"Looking at the incidence or prevalence of a disease in a population",
"mortality":	"Looking at the death rates caused by a disease",
"central line":	"a thin plastic line into a vein in the chest used for the delivery of chemotherapy",
"drug resistance": 	"It is where tumour cells become resistant to chemotherapy",
"allogeneic bmt": 	"Healthy marrow is taken from a matched donor and used to replace the patients bone marrow which has been destroyed by high dose chemotherapy",
"autologous bmt":	"In an autologous bone marrow transplant the marrow is first taken from the patient. The marrow is usually then purged with chemicals to kill any malignant cells in it",
"tumor markers":	"A substance in the body that may indicate the presence of cancer"
        }       

          
#checking if the medicine is present in the database through API's
medicine=[
        "aspirin",
"medicine",
"acetaminophen",
"tylenol",
"ibuprofen",
"advil",
"motrin IB",
"codeine",
"morphine",
"kadian",
"ms Contin",
"oxycontin",
"roxicodone",
"hydromorphone",
"dilaudid",
"oxycodone",
"exalgo",
"fentanyl",
"actiq",
"fentora",
"methadone",
"dolophine",
"methadose",
"oxymorphone",
"opana",
"ace",
"abiraterone ",
"zytiga",
"abraxane",
"abstral",
"actinomycin d",
"actiq",
"adriamycin",
"afatinib",
"giotrif",
"afinitor",
"aflibercept",
"zaltrap",
"aldara",
"aldesleukin", 
"alemtuzumab",
"mabcampath",
"alkeran",
"amsacrine",
"amsidine",
"anastrozole",
"arimidex",
"ara c",
"aredia",
"arimidex",
"panadol",
"calpol" ]      


#figuring out cancer terms
health_terms={}
for wt in word_tags:
    if wt in health_term.keys():
        health_terms.update({wt:health_term[wt]})
        
#figuring out medicine terms
medicines=[]
for wt in word_tags:
    if wt in medicine:
        medicines.append(wt)
      
                 

#Applying TF-IDF model
word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1 
        else:
            word2count[word] += 1
            
for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())
            
#Overall summary of the conversation
sent2score = {}
sent3score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 30:
                if sentence not in sent2score.keys():
                    for ht in health_terms.keys():
                        if word in ht:                
                            sent2score[sentence] = 10
                            
                            
#Medicine related summary of the conversation
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 30:
                if sentence not in sent3score.keys():
                    for m in medicines:
                        if word in m:
                            sent3score[sentence] = 10
                  
best_sentences = heapq.nlargest(5,sent2score,key=sent2score.get)
best_sentence = heapq.nlargest(5,sent3score,key=sent3score.get)

#converting list into string of cancer terms
string_health_terms = '\n'.join(health_terms)
print(health_terms)

#converting list into string of medicine prescribed
string_medicine = '\n'.join(medicines)
print(string_medicine)

#Overall Summary
string_overall_summary = ''
for sentence in best_sentences:
    string_overall_summary = string_overall_summary + sentence + '\n'
print(string_overall_summary)
    
#Medicine Summary
string_medicine_summary = ''
for sentence in best_sentence:
    string_medicine_summary = string_medicine_summary + sentence + '\n'  
print(string_medicine_summary)    
    
 
            
#SMS Notification service 
# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
#account_sid = 'AC7a2b65e083d198102faa11d2c843b8d6'
#auth_token = '79df71c72260a725c3eae0ae9a5963b6'
#summary = 'Cancer Terms: \n'+string_health_terms + '\n\n Medicine Prescribed: \n' + string_medicine+ '\n\n Overall Summary: \n'+ string_overall_summary+ '\n Medicine Summary: \n'+ string_medicine_summary
#client = Client(account_sid, auth_token)
#
#message = client.messages \
#                .create(
#                     body=summary,
#                     from_='+12019322596',
#                     to=phoneNumber
#                 )

#Email Notification
#email_from = "ekohealthsolutions@gmail.com"
#email_to = email
#message = summary
#password = "ekohealth2019"
#
#server = smtplib.SMTP('smtp.gmail.com:587')
#server.starttls()
#server.login(email_from,password)
#server.sendmail(email_from,email_to,message)
#server.quit()

