import matplotlib.pyplot as plt
proportions_dict = {'hair': 0.3137023727416992,
                    'background': 0.28617831802368165,
                    'skin': 0.2541904228210449,
                    'neck': 0.04122279510498047,
                    'cloth': 0.03395643692016601,
                    'nose': 0.02061149978637695,
                    'hat': 0.009651981353759765,
                    'l_lip': 0.006788964080810547,
                    'l_ear': 0.004702842712402343,
                    'l_brow': 0.004251541137695313,
                    'r_brow': 0.004145358276367188,
                    'u_lip': 0.004133266448974609,
                    'r_ear': 0.0038526115417480467,
                    'mouth': 0.0029729331970214843,
                    'eye_g': 0.002645770263671875,
                    'ear_r': 0.0023669219970703123,
                    'l_eye': 0.002241342926025391,
                    'r_eye': 0.0022343582153320314,
                    'neck_l': 0.000150262451171875}

sorted_labels = list(proportions_dict.keys())
sorted_proportions = list(proportions_dict.values())

plt.figure(figsize=(12, 3.6))
plt.bar(sorted_labels, sorted_proportions, color='skyblue')

plt.ylabel('Proportion of Total Pixels', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

output_pdf_path_proportion = 'proportions_given_data.pdf'
plt.tight_layout()
plt.savefig(output_pdf_path_proportion)

plt.show()
