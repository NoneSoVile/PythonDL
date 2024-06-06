import json

# Original JSON data
original_data = {
  "ancestors": [
    "android.widget.FrameLayout",
    "android.view.ViewGroup",
    "android.view.View",
    "java.lang.Object"
  ],
  "class": "com.android.internal.policy.PhoneWindow$DecorView",
  "bounds": [
    0,
    0,
    1440,
    2392
  ],
  "clickable": "false",
  "children": [
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Meme Maker",
      "bounds": [
        192,
        84,
        1440,
        217
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "New Meme",
      "bounds": [
        192,
        171,
        1440,
        258
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Top Text",
      "bounds": [
        17,
        294,
        1422,
        381
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "text": "",
      "ancestors": [
        "android.widget.TextView",
        "android.view.View",
        "java.lang.Object"
      ],
      "clickable": "true",
      "class": "android.widget.EditText",
      "bounds": [
        17,
        381,
        1422,
        591
      ],
      "componentLabel": "Input"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Bottom Text",
      "bounds": [
        17,
        608,
        1422,
        695
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "text": "",
      "ancestors": [
        "android.widget.TextView",
        "android.view.View",
        "java.lang.Object"
      ],
      "clickable": "true",
      "class": "android.widget.EditText",
      "bounds": [
        17,
        695,
        1422,
        905
      ],
      "componentLabel": "Input"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Credits",
      "bounds": [
        17,
        922,
        1422,
        1009
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "text": "",
      "ancestors": [
        "android.widget.TextView",
        "android.view.View",
        "java.lang.Object"
      ],
      "clickable": "true",
      "class": "android.widget.EditText",
      "bounds": [
        17,
        1009,
        1422,
        1149
      ],
      "componentLabel": "Input"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Font Size / Color",
      "bounds": [
        17,
        1166,
        1422,
        1253
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "ancestors": [
        "android.widget.AbsSeekBar",
        "android.widget.ProgressBar",
        "android.view.View",
        "java.lang.Object"
      ],
      "bounds": [
        17,
        1253,
        1212,
        1393
      ],
      "clickable": "false",
      "class": "android.widget.SeekBar",
      "componentLabel": "Slider"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "24",
      "bounds": [
        1229,
        1253,
        1440,
        1340
      ],
      "clickable": "false",
      "class": "android.widget.TextView",
      "componentLabel": "Text"
    },
    {
      "ancestors": [
        "android.widget.CompoundButton",
        "android.widget.Button",
        "android.widget.TextView",
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Light Font",
      "bounds": [
        35,
        1393,
        703,
        1533
      ],
      "clickable": "true",
      "class": "android.widget.RadioButton",
      "componentLabel": "Radio Button"
    },
    {
      "ancestors": [
        "android.widget.CompoundButton",
        "android.widget.Button",
        "android.widget.TextView",
        "android.view.View",
        "java.lang.Object"
      ],
      "text": "Dark Font",
      "bounds": [
        738,
        1393,
        1406,
        1533
      ],
      "clickable": "true",
      "class": "android.widget.RadioButton",
      "componentLabel": "Radio Button"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "bounds": [
        17,
        1568,
        1422,
        2392
      ],
      "clickable": "false",
      "class": "android.widget.ImageView",
      "componentLabel": "Image"
    },
    {
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "bounds": [
        0,
        84,
        175,
        259
      ],
      "clickable": "false",
      "class": "android.widget.ImageView",
      "componentLabel": "Image"
    },
    {
      "iconClass": "menu",
      "ancestors": [
        "android.view.View",
        "java.lang.Object"
      ],
      "bounds": [
        1265,
        84,
        1440,
        259
      ],
      "clickable": "false",
      "class": "android.widget.ImageView",
      "componentLabel": "Icon"
    }
  ]
}

# Function to convert original JSON data to simplified JSON format
def convert_to_simplified_json(original_data):
    simplified_data = {
        "bounds": original_data["bounds"],
        "children": []
    }
    for child in original_data["children"]:
        simplified_child = {
            "bounds": child["bounds"],
            "class": child["class"]
        }
        if "text" in child:
            simplified_child["text"] = child["text"]
        if "iconClass" in child:
            simplified_child["iconClass"] = child["iconClass"]
        simplified_data["children"].append(simplified_child)
    return simplified_data

# Convert and print simplified JSON data
simplified_data = convert_to_simplified_json(original_data)
print(json.dumps(simplified_data, indent=2))

# Save to a file
with open("simplified_data.json", "w") as file:
    json.dump(simplified_data, file, indent=2)

print("Simplified JSON data saved to simplified_data.json")
