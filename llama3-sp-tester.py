### Have transformers and peft installed using pip on the instance.
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

peft_model_id = "llama-3-sp-rallio-tester"
model_id = "unsloth/llama-3-8b"

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  torch_dtype=torch.bfloat16,
  device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

def gen_using_sp(text, temperature, num_tokens):
    encoded_input = tokenizer(text, return_tensors='pt')
    inlength = len(encoded_input['input_ids'][0])
    output_sequences = model.generate(
                                    input_ids=encoded_input['input_ids'].cuda(0),
                                    do_sample=True,
                                    max_new_tokens=num_tokens,
                                    num_return_sequences=4,
                                    top_p=0.995,
                                    temperature=temperature,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    pad_token_id=0,
                                    eos_token_id=2,
                                    use_cache=True
                                  )
    gen_sequences = output_sequences.sequences[:, :]
    responses =[]
    for sequence in gen_sequences:
        new_line=tokenizer.decode(sequence[inlength:], skip_special_tokens=True)
        responses.append(new_line)
    return responses

# Here are some prompts to test the softprompt adapter and make sure it works.
prompts = ['BestBuy, Smart Home Section: This area features a range of smart home products, from voice-controlled assistants and smart lights to advanced security systems. Interactive displays allow customers to see how these devices can be integrated into their homes. The section has a cozy, home-like atmosphere, with demonstration rooms set up to showcase the technology in a real-world setting.\nPotential',
 'BestBuy, VR Section: Virtual reality headsets and accessories are displayed prominently in this section. Customers can immerse themselves in different virtual experiences through interactive demos. Large monitors show what the users are seeing in their headsets, adding to the excitement. Staff are on hand to assist with setting up the VR equipment and to explain the features of various models.\nPotential',
 'BestBuy, Computer Accessories Section: Shelves are stocked with peripherals such as keyboards, mice, and external drives. Customers can browse a variety of brands and models, each designed to enhance their computing experience. Demonstration units allow for hands-on testing, and staff are available to answer any questions and offer suggestions based on individual requirements.\nPotential',
 'BestBuy, Accessories Section: Shelves are lined with an array of accessories, from phone cases and chargers to smartwatches and wireless earbuds. Customers browse through neatly organized rows, picking out the perfect complement to their devices. Promotional displays highlight the latest trends in tech accessories, and a few interactive kiosks provide information on compatibility and features.\nPotential',
 'BestBuy, Home Theater Section: A vast array of televisions, sound systems, and home theater setups dominate this section. Customers can experience the stunning visuals and immersive sound of the latest home entertainment technology. Reclining chairs and a dimmed lighting area provide a comfortable spot for testing out different systems. Staff are ready to offer advice and help tailor a home theater system to individual needs.\nPotential',
 'BestBuy, Entrance: The sliding glass doors open to reveal a bustling store with a sleek, modern design. Bright overhead lights illuminate rows of products, from the latest smartphones to massive flat-screen TVs. The air is filled with the hum of conversation and the occasional beeping of checkout counters. A friendly greeter at the entrance offers a warm smile and a flyer showcasing the day’s deals.\nPotential',
 'BestBuy, Mobile Phones Section: This section is a tech enthusiast’s dream, with displays of the latest smartphones from top brands. Interactive stations allow customers to test features and compare models. Knowledgeable staff members are available to answer questions and provide recommendations. The walls are adorned with posters showcasing cutting-edge phone technology and enticing upgrade offers.\nPotential',
 'BestBuy, Wearable Technology Section: Here, smartwatches, fitness trackers, and other wearable devices are showcased on sleek, modern stands. Customers can try on various models to experience their features firsthand. Large screens display promotional videos, demonstrating the capabilities and benefits of each device. Staff members are available to assist with any questions and to provide personalized recommendations.\nPotential',
 'BestBuy, Computers Section: Rows of laptops, desktops, and monitors fill this section, offering options for every type of user, from casual to professional. Interactive displays provide detailed information on specifications and features. Staff members are available to help customers find the right computer for their needs, whether it’s for gaming, work, or general use.\nPotential',
 'BestBuy, Gaming Section: This section is a paradise for gamers, with consoles, gaming PCs, and accessories all on display. Large screens showcase the latest games, and demo stations allow customers to try them out. The shelves are stocked with a wide variety of games, from the newest releases to classic favorites. Enthusiastic staff members are available to discuss game recommendations and hardware options.\nPotential']

# Test out the soft prompt.
# Output should look like below.
# Potential Actions: The player is in a particular situation. [ this action ] may result in this outcome.
for prompt in prompts:
    outputs = gen_using_sp(prompt,0.7,45)
    print("============")
    print(prompt.split("\n")[0]+"\n")
    for output in outputs:
        print("Potential"+output.split("</s>")[0]+"\n")
