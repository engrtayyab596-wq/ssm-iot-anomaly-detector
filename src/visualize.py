import matplotlib.pyplot as plt
import numpy as np


def plot_sensor_degradation(engine_data,
                            sensors, engine_id):
    fig, axes = plt.subplots(
        len(sensors), 1,
        figsize=(12, 3 * len(sensors))
    )
    for i, sensor in enumerate(sensors):
        axes[i].plot(
            engine_data['cycle'],
            engine_data[sensor]
        )
        axes[i].set_title(
            f'{sensor} — Engine {engine_id}'
        )
        axes[i].set_xlabel('Cycle')
    plt.tight_layout()
    plt.savefig(
        'sensor_degradation.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()


def plot_anomaly_scores(cycles, scores,
                        threshold, engine_id):
    plt.figure(figsize=(12, 5))
    plt.plot(cycles, scores,
             color='orange', label='Anomaly score')
    plt.axhline(
        y=threshold, color='r',
        linestyle='--',
        label=f'Threshold={threshold:.4f}'
    )
    plt.title(
        f'Anomaly Score — Engine {engine_id}'
    )
    plt.xlabel('Cycle')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(
        'anomaly_scores.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()
