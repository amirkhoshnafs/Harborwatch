# HarborWatch Class Taxonomy

## Goal
Define a compact, consistent object taxonomy for HarborWatch MVP.

The goal is not to represent every maritime object category in detail.
The goal is to create a stable, high-quality detection dataset for a real-world harbor monitoring system.

## MVP Classes
HarborWatch MVP uses exactly 3 classes:

1. `buoy`
2. `small_craft`
3. `large_vessel`

## Why only 3 classes
A compact taxonomy is better for:
- annotation consistency
- cleaner training signals
- less class confusion
- easier evaluation
- stronger portfolio presentation

This project is about building a real system well, not maximizing label complexity.

## Class Definitions

### 1) buoy
Use `buoy` for clearly visible floating navigation markers or clearly buoy-like floating markers that are relevant to the water scene.

Include:
- navigational buoys
- marker buoys
- clearly buoy-shaped floating markers

Exclude:
- reflections
- wakes
- floating debris
- tiny ambiguous specks
- dock infrastructure
- mooring ropes or anchor lines

### 2) small_craft
Use `small_craft` for watercraft that are clearly not buoys and do not appear to be large commercial / ferry / cargo-like vessels.

Include:
- small boats
- speedboats
- fishing boats
- local service boats
- compact workboats
- other clearly small-scale watercraft

Important rule:
If an object is a watercraft and is clearly not a `large_vessel`, it should usually be labeled as `small_craft`.

### 3) large_vessel
Use `large_vessel` for clearly larger maritime vessels with a more substantial profile, footprint, or structure.

Include:
- ferries
- cargo-like ships
- tanker-like ships
- larger industrial or transport vessels
- clearly large multi-deck or heavy-profile vessels

Important rule:
Use `large_vessel` only when the object clearly reads as a larger vessel class visually.
If the class is uncertain between `small_craft` and `large_vessel`, prefer consistency over overthinking and use the visual rule set in the annotation guidelines.

## Decision Order
When labeling an object, use this decision order:

1. Is it clearly a buoy or buoy-like marker?
   - Yes -> `buoy`
2. Is it clearly a watercraft?
   - No -> do not label
3. Is it clearly a large commercial / ferry / cargo / heavy-profile vessel?
   - Yes -> `large_vessel`
4. Otherwise, if it is a watercraft and not clearly large:
   - `small_craft`

## What is intentionally not a class in MVP
The following are not separate classes in MVP:
- unknown_vessel
- ship
- ferry
- kayak
- jet_ski
- tug
- sailboat
- debris
- dock
- crane
- reflection
- wake

These are merged, ignored, or excluded to preserve consistency.

## Practical Philosophy
HarborWatch is a real-world applied project.
The taxonomy is intentionally compact so the final dataset is trainable, explainable, and useful for deployment-style monitoring outputs.