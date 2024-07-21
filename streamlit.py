import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('GBMClassifier_model.pkl')

def main():
    st.title("Mental Health Diagnostic Questionnaire")

    query = {
        "Depression": [
            "1. Are you currently experiencing a depressive mood?",
            "2. Are you finding activities that you once found enjoyable less pleasurable now?",
            "3. Do you notice any changes in your thinking or memory?",
            "4. Have you noticed any changes in how you act?",
            "5. Have your sleeping or eating habits changed?",
            "6. Do you feel worthless or have low self-esteem?",
            "7. Do you have thoughts of harming yourself?",
            "8. Do you feel tired or lacking in energy?",
            "9. Do you find yourself easily getting irritated?",
            "10. Are you having trouble sleeping or sleeping too much?",
            "11. Do you find it hard to make decisions?",
            "12. Are you speaking or moving slower than usual?",
            "13. Have you noticed a decrease in your sex drive?"
        ],
        "Schizophrenia": [
            "14. Have you ever felt like something else was controlling you or influencing your actions?",
            "15. Do you sometimes believe things that others might find hard to believe?",
            "16. Do you ever see or hear things that others can't?",
            "17. Do you sometimes find it difficult to express your thoughts clearly?",
            "18. Have you noticed changes in your thinking, mood, or behavior?",
            "19. Do you have low self-esteem?",
            "20. Have you noticed a decline in your ability to do everyday tasks?",
            "21. Do you often feel like you lack motivation or interest in things?",
            "22. Does anyone in your family have mental health issues?",
            "23. Do you have any unusual movements or behaviors that others might notice?",
            "24. Do you sometimes struggle to find the motivation to do things?"
        ],
        "Acute and Transient Psychotic Disorder": [
            "25. Do you ever feel like something else was controlling you or influencing your actions?",
            "26. Do you sometimes believe things that others might find hard to believe?",
            "27. Do you ever see or hear things that others can't?",
            "28. Do you sometimes find it difficult to express your thoughts clearly?"
        ],
        "Delusional Disorder": [
            "29. Have you had any beliefs or ideas you think others might find unusual or not based in reality for at least the past three months?",
            "30. Have you experienced any symptoms typically associated with schizophrenia, such as hallucinations or disorganized thinking?",
            "31. Have you had any significant changes in your mood, such as prolonged periods of depression or episodes of extreme happiness and energy?",
            "32. Do your unusual beliefs ever go away or become less intense?",
            "33. Have you had any unusual sensory experiences, like seeing, hearing, or feeling things others do not?",
            "34. Despite your unusual beliefs, can you still maintain your work, social life, and personal care without much difficulty?"
        ],
        "Bipolar Disorder": [
            "35. Have you experienced a prolonged, heightened mood and increased activity lasting for over a week?",
            "36. Have you engaged in risky behaviors that persisted for over a week?",
            "37. Have you had rapid or flying thoughts that you couldn't control?",
            "38. Have you noticed a significant reduction in your need for sleep?",
            "39. Have you been excessively involved in activities that have a high potential for painful consequences?",
            "40. Have you been more talkative than usual?",
            "41. Have you felt an inflated sense of self-esteem or grandiosity?",
            "42. Have you experienced periods of depressed mood?",
            "43. Have you found it difficult to concentrate or been easily distracted?",
            "44. Have these symptoms occurred without being attributable to the physiological effects of substance use?",
            "45. Have you experienced severe mood disruptions?",
            "46. Have these mood disruptions lasted for longer than one week?",
            "47. Have you had an episode involving a clear change in your functioning that is uncharacteristic of you?",
            "48. Have others observed the disturbance in your mood and change in functioning?",
            "49. Has the episode lacked the severity to impair your functioning significantly or require hospitalization? If there were any psychotic features, were they during a manic episode?"
        ],
        "Anxiety Disorder": [
            "50. Do you often find yourself excessively worrying or feeling nervous?",
            "51. Have you experienced physical symptoms like a racing heart, rapid breathing, sweating, trembling, muscle tension, fatigue, dry mouth, nausea, or numbness/tingling in hands or feet?",
            "52. Do you avoid certain situations or objects because they make you feel anxious?",
            "53. Do you often feel restless or find yourself fidgeting?",
            "54. Do you have difficulty concentrating or find it hard to focus?",
            "55. Do you find yourself easily getting irritated?",
            "56. Has your anxiousness caused impairment in your social life or relationships?",
            "57. Has your anxiousness caused impairment in your work or other areas of your life?",
            "58. Have you experienced symptoms like low mood, loss of interest in activities, or feelings of hopelessness?",
            "59. Do you use alcohol or drugs as a way to cope with your anxiety?"
        ],
        "Generalized Anxiety Disorder": [
            "60. Have you experienced persistent anxiety about various aspects of your life lasting for over 6 months?",
            "61. Do you find it difficult to control your worry?",
            "62. Have you experienced excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, or sleep disturbance?",
            "63. Has your anxiety caused significant distress or impairment in your social life, work, or other areas of functioning?",
            "64. Is your anxiety not caused by substances (like drugs or alcohol) or medical conditions?",
            "65. Do your anxiety symptoms occur for most of the day and are not limited to specific situations or objects?"
        ],
        "Panic Disorder": [
            "66. Have you experienced recurrent episodes of abrupt, intense fear or discomfort, often accompanied by physical symptoms like palpitations or increased heart rate, sweating, trembling, shortness of breath, chest pain, dizziness or lightheadedness, chills, hot flushes, or fear of imminent death?",
            "67. Have you experienced physical symptoms such as palpitations or increased heart rate, sweating, trembling, shortness of breath, chest pain, dizziness or lightheadedness, chills, hot flushes, or fear of imminent death during these episodes?",
            "68. Do these episodes of intense fear or apprehension typically last for about 20-30 minutes?",
            "69. Have medical tests ruled out that these symptoms are not due to the effects of substances or medical conditions like hyperthyroidism?",
            "70. Do you worry or fear that you will have more episodes of panic attacks?"
        ],
        "Specific Phobia": [
            "71. Do you experience marked and excessive fear when exposed to specific objects or situations (e.g., blood or injury, heights, closed spaces)?",
            "72. Is the fear you experience out of proportion to the actual danger posed by the specific object or situation?",
            "73. Do you actively avoid the specific object or situation, or experience intense anxiety when confronted with it?"
        ],
        "Social Anxiety": [
            "74. Do you avoid certain objects or situations because they make you feel intensely anxious?",
            "75. Are you afraid of being negatively evaluated by others in social situations?",
            "76. Do you experience intense and persistent fear or anxiety in social situations, including conversations?",
            "77. Do you experience symptoms like blushing along with other anxiety symptoms in feared social situations?",
            "78. Do you experience symptoms like fear of vomiting along with other anxiety symptoms in feared social situations?",
            "79. Do you experience symptoms like urgency or fear of needing to urinate or defecate along with other anxiety symptoms in feared social situations?",
            "80. Do your symptoms or your avoidance of social situations cause you significant emotional distress?",
            "81. Do you recognize that your symptoms or your avoidance of social situations are excessive or unreasonable?",
            "82. Are your anxiety symptoms mainly restricted to or most prominent in the feared social situations or when thinking about them?"
        ],
        "OCD": [
            "83. Do you spend more than an hour each day dealing with obsessive thoughts?",
            "84. Do you spend more than an hour each day dealing with compulsive behaviors?",
            "85. Do you engage in ritualistic behaviors, such as washing your hands repeatedly?",
            "86. Do you find it difficult or impossible to control your obsessive thoughts?",
            "87. Have you experienced these obsessive thoughts or compulsive behaviors consistently for at least the past three weeks?"
        ],
        "PTSD": [
            "88. Have you experienced a traumatic event that caused you intense fear, helplessness, or horror?",
            "89. Have you experienced distressing symptoms after the traumatic event?",
            "90. Do you avoid people, places, or activities that remind you of the traumatic event?",
            "91. Do you avoid people, places, or activities that remind you of the traumatic event?",
            "92. Have you noticed significant changes in your thoughts or feelings since the traumatic event?",
            "93. Are you more easily startled or constantly on guard since the traumatic event?",
            "94. Do you frequently have flashbacks, dreams, or nightmares about the traumatic event?"
        ],
        "Gambling Disorder": [
            "95. Has the traumatic event significantly impacted your personal, family, social, educational, or work life?",
            "96. Do your symptoms make it difficult for you to function in your daily life?",
            "97. Do you feel restless or irritable when you try to cut down or stop gambling?",
            "98. Have you made repeated unsuccessful attempts to control, cut back, or stop gambling?"
        ],
        "Substance Abuse": [
            "99. Have you used illicit drugs in the past 12 months?",
            "100. Have you used medication that was not prescribed to you in the past 12 months?",
            "101. Have you found yourself increasing the dosage of drugs or medication to achieve the desired effect?",
            "102. Have you often used drugs or medication in larger amounts or over a longer period than you intended?",
            "103. Do you feel an uncontrollable desire or strong addiction to use drugs?",
            "104. Have you neglected important social, occupational, or recreational activities because of your drug use?",
            "105. Do you continue using drugs despite knowing that they cause harm to you?",
            "106. Do you experience withdrawal symptoms when you try to cut down or stop using drugs?",
            "107. Have you developed a tolerance to drugs, needing increasingly larger amounts to achieve the desired effect?",
            "108. Have you experienced withdrawal symptoms or a significant change in your daily functioning due to your drug use?"
        ]
    }

    responses = []
    for heading in query:
        questions = query[heading]
        st.header(heading)
        for i, question in enumerate(questions):
            st.write(question)
            response = st.radio("", ["Yes", "No"], index=1, key=f"{question}{i}")
            responses.append(1 if response == "Yes" else 0)
    
    # Add age as a dropdown with predefined ranges
    age_ranges = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 and over"]
    age = st.selectbox("What is your age range?", age_ranges)
    age_index = age_ranges.index(age)  # Convert age range to a numerical index

    # Add sex/gender as a dropdown
    sex_options = ["Male", "Female"]
    sex = st.selectbox("What is your sex/gender?", sex_options)
    sex_index = sex_options.index(sex)  # Convert sex to a numerical index

    # Append age and sex to responses
    responses.extend([age_index, sex_index])

    if st.button("Submit"):
        # Convert responses to numpy array for prediction
        responses = np.array(responses).reshape(1, -1)
        prediction = model.predict(responses)
        probabilities = model.predict_proba(responses)

        st.write("Prediction:")
        output = prediction[0]
        st.write(output)
        diagnosis_labels = {
            "Depression": output[0],
            "Schizophrenia": output[1],
            "Acute_and_transient_psychotic_disorder": output[2],
            "Delusional_Disorder": output[3],
            "BiPolar1": output[4],
            "BiPolar2": output[5],
            "Anxiety": output[6],
            "Generalized_Anxiety": output[7],
            "Panic_Disorder": output[8],
            "Specific_Phobia": output[9],
            "Social_Anxiety": output[10],
            "OCD": output[11],
            "PTSD": output[12],
            "Gambling": output[13],
            "substance_abuse": output[14]
        }
        decimals = probabilities

        labels = []
        values = []
        i = 0
        for diagnose in diagnosis_labels:
            if diagnosis_labels[diagnose] == 1:
                labels.append(diagnose)
                values.append(decimals[i][0][1])
            i += 1
        st.write(labels)
        st.write(values)
        
        st.title("Pie Chart Example")

        # Create a pie chart
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',  # Format of the percentage
            startangle=140,
            colors=plt.get_cmap('tab10').colors
        )

        # Add a legend
        ax.legend(wedges, labels,
                  title="Categories",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        # Add a title
        plt.title('Distribution of Categories')

        # Display the pie chart in Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()
