import { Component, OnInit, signal, ElementRef, ViewChildren, QueryList, AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MediaJob } from '../media-search/media-search';

@Component({
  selector:    'app-media-list',
  standalone:  true,
  imports:     [CommonModule, RouterModule],
  templateUrl: './media-list.html',
  styleUrl:    './media-list.css',
})
export class MediaList implements OnInit {
  recentJobs  = signal<MediaJob[]>([]);
  popularJobs = signal<MediaJob[]>([]);
  loading     = signal(true);
  error       = signal<string | null>(null);

  private readonly API = '/api/media/';

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get<MediaJob[]>(`${this.API}recent/`).subscribe({
      next: jobs => {
        this.recentJobs.set(jobs.slice(0, 20));
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Could not load recent searches.');
        this.loading.set(false);
      },
    });

    this.http.get<MediaJob[]>(`${this.API}popular/`).subscribe({
      next: jobs => this.popularJobs.set(jobs.slice(0, 20)),
      error: () => {},
    });
  }

  scroll(rowEl: HTMLElement, direction: 'left' | 'right') {
    const amount = 320;
    rowEl.scrollBy({ left: direction === 'right' ? amount : -amount, behavior: 'smooth' });
  }

  graphUrl(job: MediaJob): string {
    return `/media/${job.id}/graph`;
  }

  isComplete(job: MediaJob): boolean {
    return job.status === 'c';
  }

  trackById(_: number, job: MediaJob): number {
    return job.id;
  }
}
