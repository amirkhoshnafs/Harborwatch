# HarborWatch Annotation Guidelines

## Purpose
These guidelines define how HarborWatch MVP images should be labeled for object detection.

The goal is to create a consistent, deployable maritime detection dataset from raw coastal video frames.

This document is designed to reduce label noise and class drift.

## Task Type
- Task: object detection
- Output format target: COCO
- Classes:
  - `buoy`
  - `small_craft`
  - `large_vessel`

## Global Annotation Principles

### 1) Consistency is more important than perfect semantic detail
If a case is ambiguous, choose the rule that keeps the dataset consistent.

### 2) Do not invent extra classes
Only use the 3 defined HarborWatch classes.

### 3) Ignore unresolved ambiguity
If the object cannot be labeled confidently enough, do not force a label.

### 4) One physical object = one bounding box
Do not group multiple vessels into one box.

### 5) Bounding boxes should be tight and practical
Boxes should tightly cover the visible object body, without unnecessary background.

## Class-Specific Guidance

## buoy
Label as `buoy` when:
- the object is clearly a buoy or buoy-like floating marker
- the object is visible enough to box meaningfully
- it is not just a reflection or tiny speck

Boxing rule:
- include the visible buoy body / marker body
- include the visible top marker if it is clearly part of the buoy
- exclude reflection
- exclude anchor line / rope
- exclude wake-like water disturbance

Do not label as `buoy` when:
- the object is floating debris
- the object is too tiny to classify
- the object is visually unresolved

## small_craft
Label as `small_craft` when:
- the object is clearly a watercraft
- it is not a buoy
- it does not clearly read as a `large_vessel`

Examples:
- small boats
- fishing boats
- speedboats
- compact service boats
- small workboats

Boxing rule:
- include the visible vessel body and clearly attached visible superstructure
- exclude wake
- exclude reflection
- exclude detached water spray

## large_vessel
Label as `large_vessel` when:
- the object is clearly a larger maritime vessel
- it has a heavier, larger, or more substantial vessel profile
- it visually reads as a ferry / cargo-like / tanker-like / large industrial vessel

Examples:
- ferries
- cargo ships
- tanker-like ships
- large industrial vessels

Boxing rule:
- include the visible main vessel body and clearly attached superstructure
- exclude wake
- exclude reflection
- exclude detached water splash

## Tiny Object Policy

### Hard ignore
Do not label objects when the approximate bounding box would be smaller than about 8 pixels on both sides.

### Preferred annotation region
Objects are preferred for annotation when their box is at least about 12 pixels on the short side and the class is visually clear.

### Ambiguous small objects
If an object is very small and the class is not reliable, ignore it.

## Occlusion and Truncation Rules

## Truncated by image border
Label truncated objects only if:
- roughly 30 percent or more of the object is visible
- the class is still reasonably clear
- the visible part is boxable

Otherwise ignore.

## Occluded by another object or scene element
Label occluded objects only if:
- roughly 40 percent or more of the object is visible
- the class is still reasonably clear

Otherwise ignore.

## Background and Stationary Objects

### Background vessels
Label background vessels if they are clear enough and meet the visibility rules.

### Stationary vessels
Label stationary vessels if they are visible and valid objects.

Reason:
The detector should learn object presence, not motion state.
Motion logic belongs to later tracking and event layers.

## What to Exclude
Do not label:
- reflections
- wakes
- shadows
- water splash without a clear object body
- floating debris
- shoreline clutter
- docks
- cranes
- piers
- buildings
- ropes / anchor lines
- unresolved tiny specks

## Class Ambiguity Rules

### buoy vs vessel
If it is clearly a buoy-like marker -> `buoy`
If it is clearly a watercraft -> vessel class
If unresolved -> ignore

### small_craft vs large_vessel
Use `large_vessel` only when the object clearly reads as a large vessel.
Otherwise, if it is a watercraft, use `small_craft`.

This is an intentional bias to reduce class confusion.

## Bounding Box Quality Rules
A good bounding box should:
- tightly cover the visible object
- minimize unnecessary background
- not include reflection
- not include wake
- not merge multiple objects

A bad bounding box:
- is too loose
- includes large water background
- includes wake/reflection
- merges separate vessels

## Empty Images
Images with no valid objects may still be kept later as negative samples.

Do not invent labels just to avoid empty images.

## Final Annotator Rule
When unsure:
1. check whether the object is clearly one of the 3 classes
2. check whether the visible object is large enough and boxable
3. if still unclear, ignore it

Ignoring an ambiguous object is better than introducing noisy labels.