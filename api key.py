import google.generativeai as genai

genai.configure(api_key="AIzaSyB7fX_GkhCmO8o2Q0W_lGl0EuQcK_2sByo")

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Say hello")

print(response.text)