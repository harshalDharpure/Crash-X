# Qualitative Explanation Examples

Selected from 150-video test set for paper figures.


## A — Fine-tuning wins (large ROUGE-L gain over zero-shot)

### Example 1: video `001227` (moderate)
- Scores: ZS R-L=0.0, LLaVA=0.196, Greedy=0.453 (tIoU 0.0), TCD=0.503 (tIoU 0.0)

**GT**

> The weather remains normal during the evening hours, on a two-lane road, the camera car was driving on the right side of the left curved road. At the same time, a white car was approaching from the opposite direction, driving straight towards the camera car. To avoid a collision, the camera car made a left turn. Unfortunately, the front left side of the camera car collided with the front left side of the white car.

**Zero-shot Qwen2.5-VL-7B**

> **
1. **Initial Conditions

**CrashLogic Greedy**

> In the afternoon, the weather conditions remained clear, a camera car was in motion on a two-lane road. At that moment, there were only a few vehicles on the road. Suddenly, a white car appeared from the opposite direction, speeding recklessly. The white car lost control and skidded, resulting in a collision with the camera car. The impact occurred between the front of the white car and the left side of the camera car. This unfortunate incident caused damage to both vehicles.

**CrashLogic TCD α=0.5**

> In the afternoon, the weather conditions were normal as the camera car traveled along the right side of a two-way road. Simultaneously, a white car approached from the opposite direction on the left side of the same road. Unfortunately, both vehicles attempted to cross the intersection, resulting in a collision between the front-left of the camera car and the front-right of the white car.

---

### Example 2: video `000651` (moderate)
- Scores: ZS R-L=0.0, LLaVA=0.172, Greedy=0.404 (tIoU 0.0), TCD=0.449 (tIoU 0.0)

**GT**

> In the late hours of the night, the weather conditions were ordinary. The intersection is bathed in the glow of street lights. A traffic signal displays a green light, instructing vehicles to proceed. As the camera car maneuvers through the junction, it takes a right turn, coincidentally trailed by a silver car coming from the opposite direction. The silver car tries to avoid a collision by coming to a halt, while the camera car also attempts to stop but unfortunately fails to do so. The collision resulted in damages to the front of the camera car and the front right of the silver car.

**Zero-shot Qwen2.5-VL-7B**

> **

   - **

**CrashLogic Greedy**

> In the midst of a nighttime scenario, where street lights illuminate the surroundings, an intersection road becomes the setting for an incident. The traffic signal displays a green light, indicating the flow of vehicles. As the camera car proceeds along the intersection road, a black car suddenly emerges from the right side and enters the scene. Unfortunately, the black car collides with the front-right section of the camera car and the front-left portion of the black car.

**CrashLogic TCD α=0.5**

> In the midst of a nighttime scenario, the weather conditions remained clear, the glow of street lights illuminated the surroundings, revealing a spacious road. On the right side of the road, a few cars were parked. As the traffic signal turned green, the camera car began to move forward. Unexpectedly, a black car appeared from the opposite direction, attempting to cross the road. Unfortunately, the camera car was unable to stop in time, resulting in a collision between the front of the camera car and the right side of the black car.

---

### Example 3: video `000461` (moderate)
- Scores: ZS R-L=0.101, LLaVA=0.255, Greedy=0.503 (tIoU 0.0), TCD=0.458 (tIoU 1.0)

**GT**

> The weather conditions during the daytime were normal, with visible but unlit street lights due to the brightness of the day. A car was parked on the side lane, patiently waiting for the traffic signal to change. Once the signal turned green, the camera car proceeded to cross the road. However, at the same time, a black car from the right lane approached the camera car at a high speed, attempting to cross the intersection before it. Unfortunately, this unexpected chain of events resulted in a collision between the front-right of the camera car and the front-left of the black car, leading to an accident.

**Zero-shot Qwen2.5-VL-7B**

> 1. **Visibility and Weather Conditions:** The clear weather and bright sunlight might have affected the driver's ability to see clearly, especially if there were glare or reflections on the windshield or

**CrashLogic Greedy**

> In the daytime, at an intersection with a traffic light, the camera car was proceeding straight in the first lane, aiming to cross the intersection. The traffic signal displayed a green light for the camera car's direction. However, unexpectedly, another black car from the opposite direction failed to stop and attempted to cross the intersection, resulting in a collision between the front-right of the camera car and the front-left of the black car.

**CrashLogic TCD α=0.5**

> In the daytime, the weather conditions were normal, and the residential area was illuminated by several street lights. At the intersection, the traffic signal displayed a green light. The camera car proceeded to cross the intersection when, out of nowhere, a black car emerged from the right side and collided with the camera car. The collision occurred at the front of the black car and the left side of the camera car.

---

### Example 4: video `001297` (moderate)
- Scores: ZS R-L=0.023, LLaVA=0.228, Greedy=0.421 (tIoU 0.0), TCD=0.369 (tIoU 0.0)

**GT**

> The incident took place during the evening timeframe of the day. The weather condition is exceedingly regular.Numerous trees lined both sides of the road. The camera car proceeded straight ahead, while a silver car attempted to cross from the left side to the right side. Regrettably, during this maneuver, the silver car collided with a stationary white car positioned in the middle of the road. Consequently, the collision occurred between the rear of the white car and the left side of the silver car.

**Zero-shot Qwen2.5-VL-7B**

> **

1. **Initial Conditions:**

**CrashLogic Greedy**

> In the daytime, the weather conditions were clear, and the residential area was illuminated by streetlights that were not yet activated. The vehicles were parked on the right side of the one-lane road, ensuring a safe distance from the intersection. However, as the camera car approached the intersection, an unexpected incident occurred. A white car suddenly attempted to cross the intersection from the left side to the right side, while at the same time, a grey car tried to cross from the right side to the left side. This simultaneous movement led to a collision between the front of the white car and the rear of the grey car.

**CrashLogic TCD α=0.5**

> In the evening, the weather conditions were normal as cars swiftly traveled along a two-lane road. The camera car was in motion, progressing along this road. Suddenly, a white car appeared from the opposite direction, attempting to change lanes. However, it failed to notice the presence of the camera car, leading to a collision between the two vehicles. Consequently, a minor accident occurred, resulting in damage to the front-right of the camera car and the rear-left of the white car.

---


## B — Zero-shot fluent but unfaithful

### Example 1: video `000104` (moderate)
- Scores: ZS R-L=0.091, LLaVA=0.16, Greedy=0.406 (tIoU 0.25), TCD=0.306 (tIoU 0.25)

**GT**

> Numerous vehicles were in motion on a spacious road in a commercial area during daylight hours. As the camera car traveled alongside, a red bus passed by the camera car. Notably, a truck ahead of the camera car was moving at a slow pace, creating the illusion of another vehicle colliding behind the camera car. Consequently, the front end of the camera car collided with the rear end of the Black truck.

**Zero-shot Qwen2.5-VL-7B**

> **
Based on the frames provided:
- **Weather Conditions:** The overcast sky suggests cloudy weather, which could potentially reduce visibility for drivers.
- **Road Conditions:** The road appears to be clear with no visible obstacles or hazards.
- **Traffic Conditions:** Traffic seems

**CrashLogic Greedy**

> During the evening hours, the weather was overcast as the camera car followed closely behind a red car on a two-lane road. Both cars were moving slowly due to other vehicles ahead of them being stationary. Suddenly, the red car abruptly applied its brakes, causing the camera car to come to a halt as well. Unfortunately, the front of the camera car collided with the rear of the red car.

**CrashLogic TCD α=0.5**

> During the evening hours, the weather was overcast. The road was filled with a multitude of vehicles, causing a significant disruption in the flow of traffic. Among them, the camera car was also making its way along the road. On the left side of the camera car, several cars were parked, while a few pedestrians were walking on the right side. As the camera car attempted to overtake the black car in front of it, it encountered a red car that had come to a stop. Consequently, the camera car collided with the stationary vehicle, resulting in damage to the front of camera car and the rear of the red car.

---

### Example 2: video `001442` (minor)
- Scores: ZS R-L=0.116, LLaVA=0.153, Greedy=0.395 (tIoU 0.0), TCD=0.333 (tIoU 0.0)

**GT**

> In a misty afternoon, on a narrow two-way expressway, the camera car proceeded straight in its designated lane. Ahead of the camera car, a grey SUV attempted to change lanes in order to pass the purple bus in front of it. However, a white car suddenly appeared, the grey SUV was unable to brake in time.then the left side of the grey SUV collided with the left side of the white car.

**Zero-shot Qwen2.5-VL-7B**

> **

1. **Initial Conditions:**
   - The road appears to be dry, and there are no visible signs of wetness or slippery conditions that could have contributed to the crash.
   - The visibility is slightly reduced due to the overcast weather, but this alone does not seem to be a primary

**CrashLogic Greedy**

> In the midst of a misty evening, on a two-lane road, the camera car was proceeding straight in its designated lane. At the same time, a white car was approaching from the opposite direction, while a large blue bus was also traveling ahead of the camera car. Suddenly, an unannounced left turn of the blue bus caught the camera car off guard, leading to a collision between the front-left of the camera car and the rear-right of the blue bus.

**CrashLogic TCD α=0.5**

> In the midst of a misty evening, a two-lane road was partially obscured by a dense fog. The camera car was in motion on this road, with a white car ahead and a bus approaching from the opposite direction. As the bus made a right turn, it collided with the white car, causing the camera car to be affected by the impact on its front left side and the rear right side of the bus.

---

### Example 3: video `000661` (n/a)
- Scores: ZS R-L=0.117, LLaVA=0.109, Greedy=0.276 (tIoU 0.0), TCD=0.314 (tIoU 0.0)

**GT**

> Under the pleasant and normal weather, the highway road was bustling with numerous vehicles in motion. Among them, the camera car was also moving steadily. On the far right side, a white truck was making its way. In the meantime, a red car made an attempt to overtake the white truck, but fortunately, no accidents took place.

**Zero-shot Qwen2.5-VL-7B**

> ### Analysis of the Video

The provided video appears to be a dashcam recording from inside a vehicle traveling on a multi-lane highway. The scene unfolds as follows:

1. **Initial Conditions:**
   - The vehicle is traveling in the right lane.
   - There are multiple vehicles ahead, including a red sedan directly in front of the camera vehicle.

2. **Incident Description:**
   - At approximately 0:05 seconds into the video, there is a sudden jolt forward, indicating a collision or a significant impact.
   - The vehicle in front of the camera vehicle (the red sedan) appears to have been involved in an accident, as it is seen skidding and coming to a stop abruptly.
   - The camera vehicle itself seems to have been pushed forward by the impact, as indicated by the forward movement of the dashboard and the position of the camera.

3. **Crash Severity:**
   - The severity of the crash appears to be moderate to severe. The sudden forward movement suggests that the vehicle was hit from behind with enough force to cause noticeable damage and displacement.
   - The red sedan's skid marks and the abrupt stop suggest that the impact was strong enough to cause significant damage to both vehicles involved.

4. **Impact

**CrashLogic Greedy**

> During daylight hours, the weather was clear, and there were no street lights on the one-lane road where vehicles were traveling at a high speed. A camera car was also moving along this road. At that moment, a white truck was traveling in the opposite direction of the camera car, while a red car was moving ahead of it. Suddenly, the red car made a left turn without noticing the presence of the white truck. As a result, the white truck collided with the rear left side of the red car, causing damage to both vehicles. The impact primarily occurred between the front-right of the white truck and the rear left of the red car.

**CrashLogic TCD α=0.5**

> During daylight hours, the weather was clear, and the road was filled with numerous street lights. It was a four-lane road, and the camera car was positioned in the middle lane, following a red car. On the opposite side, several vehicles were also in motion. In this particular scenario, a white car attempted to overtake the red car but unexpectedly collided with it. The impact occurred between the front right of the white car and the rear left of the red car.

---

### Example 4: video `000557` (severe)
- Scores: ZS R-L=0.123, LLaVA=0.192, Greedy=0.398 (tIoU 0.5), TCD=0.279 (tIoU 0.5)

**GT**

> In a climate characterized by rainfall, the roads become wet. On a highway, vehicles are constantly in motion. On the right side of the camera car, both an emergency vehicle and another car have come to a halt for unknown reasons. Ahead of the camera vehicle, a white truck is traveling along a road, while in front of the white truck, a black truck is slowing down. Unfortunately, the white truck almost crashes into the black truck, resulting in a collision between the front right of the white truck and the rear left of the black truck.

**Zero-shot Qwen2.5-VL-7B**

> **

1. **Weather Conditions:** The video shows wet road conditions, likely due to rain, which can reduce traction and increase the risk of accidents. Wet roads make it more difficult for drivers to maintain control of their vehicles, especially at higher speeds.

2. **Speed and Distance:** The vehicles appear to

**CrashLogic Greedy**

> During the daytime in a rainy condition, the four-lane road became completely saturated, resulting in an extremely slippery surface. The camera car was traveling along the highway Road. Numerous vehicles were parked on the right side of the road. The white truck was moving at a slow pace due to the numerous vehicles halted in front of it. The grey car, which was also on the highway, skidded and collided with the white truck. The collision occurred between the front right of the grey car and the rear left of the white truck.

**CrashLogic TCD α=0.5**

> During the rainy season, on a two-lane road, a scenario unfolds where a black car is traveling at speed on the right side of the road. Simultaneously, a grey car is approaching from the opposite direction. The black car loses control and veers towards the left side of the road, colliding with the grey car. The impact occurs between the front of the black car and the rear of the grey car. This incident suggests a loss of control by the black car, resulting in a collision with the grey car on the rainy two-lane road. Investigations into the incident may consider factors such as road conditions, speed, and the actions of both drivers to determine the cause and responsibility for the collision.

---


## C — TCD improves temporal consistency vs Greedy

### Example 1: video `000025` (moderate)
- Scores: ZS R-L=0.18, LLaVA=0.259, Greedy=0.252 (tIoU 0.0), TCD=0.325 (tIoU 1.0)

**GT**

> In snowy weather on a residential area road, a scenario unfolds where a red car is traveling on the right side of the road ahead of a car equipped with a camera. The camera car attempts to overtake the red car by moving to the left side of the road. However, suddenly, the red car changes its course to the left side of the road and collides with the camera car. As a result of the collision, the camera car goes off-road and crashes into the deposited snow. The impact occurs between the front right corner of the camera car and the front left corner of the red car. This incident suggests a sudden change in the red car's course, leading to a collision with the overtaking camera car, resulting in the camera car leaving the road and encountering the snowy area. Investigations into the incident may consider factors such as signaling, visibility, and the actions of both drivers to determine the cause and responsibility for the collision in the snowy residential area.

**Zero-shot Qwen2.5-VL-7B**

> The provided video appears to be a dashcam recording of a vehicle driving on a snowy road during the day. The timestamp in the video indicates that it was recorded on December 22, 2012, at 13:48:07.

### Analysis:

1. **Crash Severity**: 
   - The video does not show any immediate evidence of a crash occurring within the frame. There is no visible damage to the vehicle or any other vehicles involved.
   - The video seems to capture a moment where the vehicle is driving normally until it encounters an obstacle or another vehicle.

2. **Impact Point**:
   - The impact point cannot be determined from the provided video as there is no visible collision or damage. The vehicle continues to drive forward without any abrupt changes in direction or speed.

3. **Timestamp Window**:
   - The timestamp provided (13:48:07) is the exact moment when the video was recorded. Since the video does not show a crash, the timestamp window for the crash would be before this moment.

4. **Causal Explanation**:
   - Based on the limited information available in the video, it is impossible to determine the cause of any potential crash or incident. The video only shows

**CrashLogic Greedy**

> In the midst of a snowy weather condition, a two-lane road was partially covered with snow on its sides. The camera car was in motion, navigating through the road. Suddenly, a black car appeared from the opposite direction, sliding towards the side of the road due to the slippery surface caused by the wet conditions. Unfortunately, the front of the camera car collided with the rear of the black car.

**CrashLogic TCD α=0.5**

> In the midst of a snowy weather, a two-lane road was partially covered with a layer of snow on its sides. The camera car was in motion, swiftly traveling along the road. At the intersection, a red car attempted to enter the main road, while a black car approached from the right side of the intersection, intending to cross the main road. Regrettably, due to the slippery surface, the black car skidded towards the camera car. Consequently, a collision occurred between the front of the camera car and the front left of the black car.

---

### Example 2: video `000051` (moderate)
- Scores: ZS R-L=0.153, LLaVA=0.241, Greedy=0.252 (tIoU 0.0), TCD=0.282 (tIoU 1.0)

**GT**

> On a wide city road during the night with illuminated surroundings, vehicles are moving at speed. The camera car records an incident as all vehicles approach a sharp diversion. After taking the diversion, a car traveling in the middle of the road loses control and collides with another car. The impact occurs between the front left corner of the first car and the front right corner of the other car. This scenario suggests that the black car's loss of control may have been a contributing factor leading to the collision after negotiating the sharp diversion on the wide city road at night. Investigations into the incident may consider factors such as road conditions, speed, and the actions of the first car's driver to determine the cause and responsibility for the collision.

**Zero-shot Qwen2.5-VL-7B**

> ### Analysis of the Video

The provided video appears to be a dashcam recording taken at night in an urban environment. The timestamp on the video indicates that it was recorded on February 25, 2013, at 23:24:38.

#### Observations:
- **Environment:** The scene is a busy street with multiple lanes of traffic, illuminated by streetlights and building lights.
- **Vehicles:** There are several vehicles visible, including cars and what seems to be a bus or large vehicle in the background.
- **Activity:** The vehicles appear to be moving at a moderate speed, typical for city traffic at night.

#### Potential Crash Scenario:
Based on the frames provided, there doesn't seem to be a clear indication of a car crash occurring within these specific frames. However, if we assume that a crash did occur, here's a possible analysis:

1. **Timestamp Window:** The exact timestamp window where the crash might have occurred would need to be identified from the video timeline. Since the provided timestamp is 23:24:38, we can infer that the crash could have happened around this time.

2. **Impact Point:** If a crash did occur, the impact point would likely be on one of

**CrashLogic Greedy**

> In the midst of a nighttime scenario, where street lights illuminate the surroundings, an intersection road finds itself teeming with a multitude of vehicles in motion. Among them, a camera car gracefully traverses the road, while a black car and a white car come to a halt on the right side. Suddenly, without any prior warning, the black car makes an attempt to collide with the white car. However, the white car swiftly veers to the left, resulting in a collision between the front right of the black car and the front left of the white car.

**CrashLogic TCD α=0.5**

> In the midst of a nighttime scenario, the weather conditions remained clear, the glow from streetlights illuminated the surroundings, revealing only a few vehicles in motion on a four-lane road. It is a commercial area. Unexpectedly, a black car swiftly overtook the grey car and collided with it. The collision occurred between the front right of the black car and the front left of the grey car, resulting in moderate damage to both vehicles. The camera car was following the black car in the same lane.

---

### Example 3: video `000077` (moderate)
- Scores: ZS R-L=0.174, LLaVA=0.194, Greedy=0.363 (tIoU 0.0), TCD=0.389 (tIoU 1.0)

**GT**

> During the evening hour, In the midst of a snowy weather situation, a four-lane road becomes wet and partially covered with snow along its edges.The vehicle comes to a halt as the camera car observes the red traffic signal. At a traffic light intersection, a series of occurrences takes place. The traffic light shows a red signal, while pedestrians are crossing the road. All of a sudden, a black car, appearing to be traveling at a significant speed, crashes into the moving silver car as it tries to enter the opposite side of the road. The collision happens between the front left corner of the black car and the front right corner of the silver car.

**Zero-shot Qwen2.5-VL-7B**

> Given the lack of visible evidence of a crash, the image likely captures a moment before or after a potential incident. It shows a rainy street with pedestrians crossing the road and vehicles moving along. The weather conditions suggest that visibility might be reduced due to rain, which could increase the risk of accidents if drivers are not cautious. However, without additional frames or context, it

**CrashLogic Greedy**

> In the midst of snowy weather, an incident occurred at a four-way intersection where all lane roads were covered in snow. The road was wet, and several vehicles were in motion. Pedestrians were walking on the sidewalk when suddenly, a black car emerged from the right side of the road, attempting to cross the intersection. At the same time, the camera car was also traveling in its direction. Unfortunately, due to the slippery surface, the black car skidded and collided with the front-left of the camera car and the right-side of the black car.

**CrashLogic TCD α=0.5**

> In the midst of snowy weather, a two-lane road, partially covered with snow, served as the backdrop for a pedestrian crossing the road at an intersection. Simultaneously, the intersection was illuminated by a bright light, and a traffic signal displayed a green light. The camera car proceeded along the road, while a black car approached from the opposite direction. Unfortunately, the black car failed to notice the presence of the camera car, leading to a collision between the two vehicles. Consequently, the camera car sustained damage to its front right side, while the left side of the black car was also affected.

---

### Example 4: video `000311` (minor)
- Scores: ZS R-L=0.0, LLaVA=0.182, Greedy=0.318 (tIoU 0.0), TCD=0.349 (tIoU 1.0)

**GT**

> The day is transitioning into evening time, the weather is ordinary, devoid of any significant changes. A well-maintained and dry two-lane road provided a clearly visible route. As it was daytime, the streetlights remained unilluminated. The yellow construction vehicle in front of the camera car was being operated carefully, staying within the correct lane and following the assigned route. Several vehicles were parked along the sides of the road. Unexpectedly, a black car which was parked on the right side of the road began to move towards the left side in order to make a turn, resulting in an unfortunate collision with the yellow construction vehicle. This collision took place at the front-right corner of the yellow construction vehicle and the rear-left corner of the black car.

**Zero-shot Qwen2.5-VL-7B**

> **

   - **Possible

**CrashLogic Greedy**

> In the daytime, the weather conditions remained clear, with numerous vehicles parked along the roadside. The camera car was in motion on the road, capturing the movement of other vehicles. As the white car attempted to overtake the camera car, an unexpected incident occurred. Simultaneously, the grey van also tried to pass the camera car. Unfortunately, due to the limited distance between them, a collision took place between the front right of the white car and the rear left of the grey van.

**CrashLogic TCD α=0.5**

> During daylight hours, the weather conditions remained clear, and the road was devoid of parked vehicles. The camera car was in motion, capturing the activity on the road. At that moment, a pedestrian was attempting to cross the road. Meanwhile, on the opposite side of the road, a black car was also in motion. Unexpectedly, an orange construction vehicle emerged from the opposite direction, losing control and colliding with the black car. Consequently, the front of the construction vehicle collided with the rear of the black car.

---


## D — Remaining hard cases (limitations)

### Example 1: video `001337` (severe)
- Scores: ZS R-L=0.067, LLaVA=0.204, Greedy=0.184 (tIoU 1.0), TCD=0.254 (tIoU 1.0)

**GT**

> During the rainy evening and the road is wet. At the junction, vehicles were crossing the road. A camera car was moving straight ahead on the main road while four mopeds were stopped in front of it due to a red traffic light. On the opposite lane, below the bridge, a green van occupied the right lane, while directly across from it was a white truck. Both the green van and white truck continued straight towards the camera car. However, the green van made a left turn without noticing the white truck, resulting in a collision. The front-left of the green van collided with the front-right of the white truck, causing damage to the front sides of both vehicles. Consequently, the white truck swerved towards the white moped and the red moped, causing the right side of the white moped to fall onto the left side of another red moped. Additionally, the white truck crashed into the central median divider.

**Zero-shot Qwen2.5-VL-7B**

> **

Based on the visual evidence from the video:

- **Scenario:** The video shows multiple vehicles stopped under an overpass,

**CrashLogic Greedy**

> Amidst the rainy season, the road became wet as an ant hill. A bustling intersection was filled with a multitude of vehicles, all patiently waiting for the signal to change. Among them, three motorcyclists were making their way along the road when suddenly, one of them lost control of their bike and collided with another bike. The bike that was affected was the yellow one, and it collided with the rear left side of a white bike. This unfortunate incident unfolded right in front of the camera car.

**CrashLogic TCD α=0.5**

> Amidst the rainy season, the road became wet, resulting in a slippery surface. The intersection was bustling with a multitude of vehicles, while the pedestrians patiently waited on the sidewalk, their attention directed towards the intersection. The camera car came to a halt, awaiting the signal to change. Regrettably, the yellow bike suddenly applied its brakes, leading to an unfortunate collision with the camera car. In the midst of the chaos, a white bus approached the intersection, while a yellow taxi traveled along the same road. The collision involved the front of the yellow taxi, rear of the white bus, and front of the yellow bike.

---

### Example 2: video `000894` (moderate)
- Scores: ZS R-L=0.0, LLaVA=0.198, Greedy=0.195 (tIoU 0.0), TCD=0.283 (tIoU 0.5)

**GT**

> On a two-lane, two-way road, the car equipped with a camera is traveling straight on the right side, while another black car approaching from the opposite direction switches lanes from the left lane to the right lane. This maneuver leads to a collision between the front left part of the black car and the front left part of the car with the camera view, causing the car with the camera view to veer off the road eventually.

**Zero-shot Qwen2.5-VL-7B**

> **
   - **

**CrashLogic Greedy**

> In the daytime, under clear weather conditions, there are no parked vehicles or street lights in sight. The only vehicles on the road are a white van and a camera heavy vehicle, with no pedestrians in sight. Suddenly, the white van attempts to turn onto the service line but unfortunately loses control and skids into the opposite lane. Tragically, the front of the camera heavy vehicle crashes into the right side of the white van.

**CrashLogic TCD α=0.5**

> During the daytime, the weather was clear, and the road was filled with numerous street lights that remained unlit. The road stretched out wide, with green grass lining both sides. As the camera car advanced along the highway, it encountered several other vehicles. Unexpectedly, the black car abruptly veered left, resulting in a loss of control and a subsequent collision between the camera car and the black car. The impact caused the camera car to spin and eventually come to a halt on the grassy area. Consequently, a collision occurred between the front of the camera car and the left side of the black car.

---

### Example 3: video `001438` (n/a)
- Scores: ZS R-L=0.1, LLaVA=0.126, Greedy=0.201 (tIoU 0.0), TCD=0.224 (tIoU 0.0)

**GT**

> As the evening descended, the road lay stretched out before us, illuminated by a clear sky. The edges of the road were blanketed with snow. The camera car was making its way along the highway, trailing behind the gray car. unfortunately , no accidents occurred in this specific area, bringing about a sense of relief.

**Zero-shot Qwen2.5-VL-7B**

> **
   - **Weather Conditions:** The presence of snow

**CrashLogic Greedy**

> In the midst of a snowy weather condition, an incident occurred on a two-lane road. The camera car was steadily moving forward on the main road. Surrounded by trees on both sides, the road was partially covered with snow. Suddenly, a black car overtook the camera car and continued speeding straight ahead. At the same time, a red car approached from the opposite direction and attempted to make a turn. Unfortunately, the red car failed to notice the presence of the black car and ended up colliding with it. As a result, a collision took place between the front-left side of the black car and the rear-right side of the red car.

**CrashLogic TCD α=0.5**

> In the midst of the wintry weather, a two-lane road, partially covered with snow, served as the backdrop for an incident. The camera car was steadily moving forward on the main road. Surprisingly, a black car overtook the camera car and continued its journey. However, due to the slippery surface, the black car lost control and swerved towards the right side of the road, attempting to collide with the red car. Regrettably, the front right of the black car collided with the rear left of the red car.

---

### Example 4: video `000492` (n/a)
- Scores: ZS R-L=0.162, LLaVA=0.201, Greedy=0.218 (tIoU 0.0), TCD=0.201 (tIoU 0.0)

**GT**

> Despite the snow-covered sides of the road, the path remained clear for the vehicles, the snow is melting which is causing the wet road. In the evening hours, within a bustling commercial district surrounded by towering buildings, a cluster of cars patiently awaited their turn in the traffic. The camera car steadily progressed along the main road. Among them, the driver of the white car signaled a left turn. Simultaneously, a grey van on the left side also maneuvered towards the left, narrowly avoiding a minor collision as both vehicles swiftly escaped the potential accident.

**Zero-shot Qwen2.5-VL-7B**

> **
Based on the visual cues:
1. **Speed and Momentum:** The vehicles appear to be moving at a moderate speed, given the distance between them and the presence of other vehicles in the background.
2. **Visibility Conditions:** The weather seems overcast, which could affect visibility for drivers, potentially contributing to the accident.
3.

**CrashLogic Greedy**

> In the evening, amidst snowy weather conditions, the road was bustling with numerous vehicles traversing their designated paths on a four-lane road. The camera car was steadily moving forward on the main road. Among them, a black car was traveling along the third lane, while a white bus occupied the fourth lane. Both vehicles were in close proximity to each other when, unexpectedly, the black car lost control and collided with the rear of the white bus. The collision caused moderate damage to the front of the black car and the rear of the white bus.

**CrashLogic TCD α=0.5**

> In the evening, amidst snowy weather, the road was covered in a layer of snow. The residential area was illuminated by several street lights, and the four-lane road was relatively calm, with only a few vehicles in motion. One of these vehicles was the camera car, which was moving at a moderate speed. Suddenly, a black car appeared from the opposite direction, speeding recklessly. Despite the presence of other vehicles, the black car failed to control its speed and collided with the white bus, which was traveling in the same lane. The impact occurred at the front of the black car and the rear of the white bus, resulting in moderate damage.

---
