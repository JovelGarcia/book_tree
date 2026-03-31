import { Component, OnInit, signal } from '@angular/core';
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
  jobs    = signal<MediaJob[]>([]);
  loading = signal(true);
  error   = signal<string | null>(null);

  private readonly API = '/api/media/';

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get<MediaJob[]>(`${this.API}recent/`).subscribe({
      next: jobs => {
        this.jobs.set(jobs.slice(0, 5));
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Could not load recent searches.');
        this.loading.set(false);
      },
    });
  }

  statusClass(s: MediaJob['status']): string {
    return `badge badge--${s}`;
  }

  graphUrl(job: MediaJob): string {
    return `/epubs/${job.id}/relationships`;
  }

  trackById(_: number, job: MediaJob): number {
    return job.id;
  }
}
