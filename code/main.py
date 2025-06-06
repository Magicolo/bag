from dataclasses import dataclass
from typing import Sequence, Set, Tuple

from cv2 import circle, line
from cv2.typing import MatLike, Scalar
from audio import Audio
from camera import Camera
import vector
from detect import Detector, Hand, Landmark, Pose
import measure
from window import Window


_PLAYERS = 2


@dataclass
class Player:
    pose: Pose
    hands: Tuple[Hand, Hand]


def draw(
    frame: MatLike,
    players: Sequence[Player],
    hands: Sequence[Hand],
    poses: Sequence[Pose],
) -> MatLike:
    def draw_landmarks(
        frame: MatLike,
        landmarks: Sequence[Landmark],
        connections: Sequence[Tuple[int, int]],
        color: Scalar,
        scale: int = 1,
    ):
        height, width, _ = frame.shape
        for landmark in landmarks:
            frame = circle(
                frame,
                (int(landmark.x * width), int(landmark.y * height)),
                scale * 2,
                color,
                -1,
            )
        for connection in connections:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            frame = line(
                frame,
                (int(start.x * width), int(start.y * height)),
                (int(end.x * width), int(end.y * height)),
                color,
                scale,
            )
        return frame

    with measure.block("Draw"):
        for player in players:
            for hand in player.hands:
                frame = draw_landmarks(
                    frame, hand.landmarks, Hand.CONNECTIONS, (0, 255, 0), 2
                )
            frame = draw_landmarks(
                frame, player.pose.landmarks, Pose.CONNECTIONS, (0, 255, 0), 2
            )

        for hand in hands:
            frame = draw_landmarks(frame, hand.landmarks, Hand.CONNECTIONS, (255, 0, 0))
        for pose in poses:
            frame = draw_landmarks(frame, pose.landmarks, Pose.CONNECTIONS, (255, 0, 0))

        return frame


def update(players: Sequence[Player], hands: Sequence[Hand], poses: Sequence[Pose]):
    with measure.block("Update"):
        hand_indices: Tuple[Set[int], Set[Tuple[int, int]]] = set(), set()
        hand_distances = sorted(
            (
                (p, o, n, old.distance(new, square=True))
                for p, player in enumerate(players)
                for o, old in enumerate(player.hands)
                for n, new in enumerate(hands)
                if new.handedness == old.handedness
            ),
            key=lambda pair: (pair[3], pair[0]),
        )
        for p, o, n, _ in hand_distances:
            if n in hand_indices[0]:
                continue
            else:
                hand_indices[0].add(n)
                hand_indices[1].add((p, o))

            player = players[p]
            old = player.hands[o]
            new = hands[n]
            if o == 0:
                player.hands = (old.update(new), player.hands[1])
            else:
                player.hands = (player.hands[0], old.update(new))

        pose_indices: Tuple[Set[int], Set[int]] = set(), set()
        pose_distances = sorted(
            (
                (p, n, player.pose.distance(pose, square=True))
                for p, player in enumerate(players)
                for n, pose in enumerate(poses)
            ),
            key=lambda pair: (pair[2], pair[0]),
        )
        for p, n, _ in pose_distances:
            if n in pose_indices[0]:
                continue
            else:
                pose_indices[0].add(n)
                pose_indices[1].add(p)

            players[p].pose = players[p].pose.update(poses[n])

        for p, player in enumerate(players):
            if p in pose_indices[1]:
                for o, hand in enumerate(player.hands):
                    if (p, o) in hand_indices[1]:
                        continue
                    else:
                        hand_indices[1].add((p, o))

                    motion = vector.subtract(
                        player.pose.wrists[o].position, hand.wrist.position
                    )
                    if o == 0:
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
    players = tuple(
        Player(Pose.DEFAULT, (Hand.LEFT, Hand.RIGHT)) for _ in range(_PLAYERS)
    )
    for index, (frame, time) in enumerate(camera.frames()):
        hands, poses = detector.detect(frame, time)
        update(players, hands, poses)

        if show:
            frame = draw(frame, players, hands, poses)

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
