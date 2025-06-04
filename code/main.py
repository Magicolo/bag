from dataclasses import dataclass
from typing import ClassVar, Iterable, List, Sequence, Set, Tuple

from cv2 import circle, line, rectangle
from cv2.typing import MatLike, Scalar
from audio import Audio
from camera import Camera
import vector
from detect import Detector, Hand, Pose
import measure
from window import Window


_PLAYERS = 2


@dataclass
class Player:
    DEFAULT: ClassVar["Player"]

    pose: Pose
    hands: Tuple[Hand, Hand]


Player.DEFAULT = Player(Pose.DEFAULT, (Hand.DEFAULT, Hand.DEFAULT))


def _draw(
    frame: MatLike,
    hands: Sequence[Hand],
    poses: Sequence[Pose],
) -> MatLike:
    def _draw_landmarks(
        frame: MatLike,
        landmarks: Iterable[List[Tuple[float, float, Scalar]]],
        connections: Iterable[Tuple[int, int, Scalar]],
    ):
        height, width, _ = frame.shape
        for group in landmarks:
            for landmark in group:
                frame = circle(
                    frame,
                    (int(landmark[0] * width), int(landmark[1] * height)),
                    2,
                    landmark[2],
                    -1,
                )
            for connection in connections:
                start = group[connection[0]]
                end = group[connection[1]]
                frame = line(
                    frame,
                    (int(start[0] * width), int(start[1] * height)),
                    (int(end[0] * width), int(end[1] * height)),
                    connection[2],
                    1,
                )
        return frame

    with measure.block("Draw"):
        height, width, _ = frame.shape
        for hand in hands:
            low, high = vector.scale(hand.minimum, hand.maximum, 1.25)
            frame = rectangle(
                frame,
                (int(low[0] * width), int(low[1] * height)),
                (int(high[0] * width), int(high[1] * height)),
                (127, 127, 0),
                1,
            )
            for finger in hand.fingers:
                low, high = vector.scale(finger.minimum, finger.maximum, 1.25)
                frame = rectangle(
                    frame,
                    (int(low[0] * width), int(low[1] * height)),
                    (int(high[0] * width), int(high[1] * height)),
                    (0, 127, 127),
                    1,
                )

            for pose in poses:
                for palm in pose.palms:
                    low, high = vector.overlap(hand.bound, palm.bound)
                    if vector.area(low, high) > 0:
                        frame = rectangle(
                            frame,
                            (int(low[0] * width), int(low[1] * height)),
                            (int(high[0] * width), int(high[1] * height)),
                            (0, 0, 0),
                            1,
                        )

        frame = _draw_landmarks(
            frame,
            (
                [(landmark.x, landmark.y, (0, 255, 0)) for landmark in hand.landmarks]
                for hand in hands
            ),
            ((start, end, (0, 255, 0)) for start, end in Hand.CONNECTIONS),
        )
        frame = _draw_landmarks(
            frame, ([(hand.x, hand.y, (255, 0, 0))] for hand in hands), []
        )
        frame = _draw_landmarks(
            frame,
            (
                [(finger.x, finger.y, (255, 0, 0)) for finger in hand.fingers]
                for hand in hands
            ),
            [],
        )
        frame = _draw_landmarks(
            frame,
            (
                [(landmark.x, landmark.y, (0, 0, 255)) for landmark in pose.landmarks]
                for pose in poses
            ),
            ((start, end, (0, 0, 255)) for start, end in Pose.CONNECTIONS),
        )
        return frame


def update(players: Sequence[Player], hands: Sequence[Hand], poses: Sequence[Pose]):
    # TODO: Does this work?
    hand_indices: Tuple[Set[int], Set[Tuple[int, int]]] = set(), set()
    hand_distances = sorted(
        (
            (p, o, n, old.distance(new, square=True))
            for p, player in enumerate(players)
            for o, old in enumerate(player.hands)
            for n, new in enumerate(hands)
            # TODO: Do I really make this check?
            if new.handedness == old.handedness
        ),
        key=lambda pair: pair[3],
    )
    for p, o, n, _ in hand_distances:
        if n in hand_indices[0]:
            continue
        else:
            hand_indices[0].add(n)
            hand_indices[1].add((p, o))

        player = players[p]
        new = hands[n]
        if o == 0:
            old = player.hands[0]
            player.hands = (old.update(new), player.hands[1])
        else:
            old = player.hands[1]
            player.hands = (player.hands[0], old.update(new))

    pose_indices: Tuple[Set[int], Set[int]] = set(), set()
    pose_distances = sorted(
        (
            (o, n, old.pose.distance(new, square=True))
            for o, old in enumerate(players)
            for n, new in enumerate(poses)
        ),
        key=lambda pair: pair[2],
    )
    for o, n, _ in pose_distances:
        if n in pose_indices[0]:
            continue
        else:
            pose_indices[0].add(n)
            pose_indices[1].add(o)

        player = players[o]
        new = poses[n]
        player.pose = new.update(player.pose)

    for p, player in enumerate(players):
        if p in pose_indices[1]:
            for h, hand in enumerate(player.hands):
                if (p, h) in hand_indices[1]:
                    continue
                else:
                    hand_indices[1].add((p, h))

                motion = vector.subtract(
                    player.pose.wrists[h].position, hand.wrist.position
                )
                if h == 0:
                    player.hands = (hand.move(motion), player.hands[1])
                else:
                    old = player.hands[1]
                    player.hands = (player.hands[0], hand.move(motion))


with Audio() as audio, Camera() as camera, Window() as window, Detector(
    hands=_PLAYERS * 2, poses=_PLAYERS
) as detector:
    success = True
    frame = None
    show = False
    mute = False
    players = tuple(Player.DEFAULT for _ in range(_PLAYERS))
    for index, (frame, time) in enumerate(camera.frames()):
        hands, poses = detector.detect(frame, time)
        update(players, hands, poses)

        if show:
            frame = _draw(frame, hands, poses)

        reset = False
        key, change = window.show(frame)
        if change:
            if key == ord("d"):
                show = not show
            elif key == ord("r"):
                reset = True
            elif key == ord("m"):
                mute = not mute
            elif key in (ord("q"), 27):
                break

        audio.send(hands, poses, mute, reset)

        if index % 10 == 0:
            measure.flush()
